from typing import Literal, Optional

import pandas as pd
import timm
import torch
import torchmetrics
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics.functional.classification import accuracy, auroc

from models.preact_resnet import PreActResNet50


class WsiClassifier(LightningModule):
    def __init__(
        self,
        model: str = "resnet50",
        lr: float = 0.001,
        num_classes: int = 2,
        ckpt_path: Optional[str] = None,
        imagenet_pretrained: bool = False,
        finetune: bool = False,
        criterion: Literal["crossentropy"] = "crossentropy",
        log_params: bool = False,
        batch_size: int = 128,
        **kwargs,
    ):
        """
        Args:
            model: backbone model, either a timm model or "preact_resnet50"
            lr: learning rate
            num_classes: number of target classes
            ckpt_path: path to pretrained backbone checkpoint
            imagenet_pretrained: use imagenet pretrained weights for the backbone
            finetune: whether or not to finetune the backbone
            criterion: loss function to use
            log_params: debug: log model parameters and gradients to wandb
            batch_size: batch size
        """
        super().__init__()

        self.save_hyperparameters()

        self.backbone, self.classifier = self._init_model(
            model, num_classes, ckpt_path, imagenet_pretrained, finetune
        )

        # TODO: support for more loss functions, balance weighting, etc
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.num_classes = num_classes
        self.log_params = log_params

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def forward_features(self, x):
        return self.backbone(x)

    def on_fit_start(self):
        if self.log_params:
            self.logger.watch(self, log="all")

    def training_step(self, batch, batch_idx):
        x = batch["patch"]
        y = batch["label"]
        # x = batch["Data"]
        # y = batch["Target"].squeeze(1)
        loss, preds, scores = self.shared_step(x, y)

        self.train_acc(preds, y)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "train/patch_acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        return {"loss": loss, "patch_preds": preds, "patch_scores": scores, "y": y}

    def training_epoch_end(self, outputs):
        scores = torch.cat([x["patch_scores"] for x in outputs], dim=0).detach().cpu()
        y = torch.cat([x["y"] for x in outputs], dim=0).cpu()

        if self.num_classes == 2:
            self.log(
                "train/patch_auc",
                auroc(scores[:, 1], y, task="binary"),
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "train/patch_auc",
                auroc(scores, y, task="multiclass"),
                prog_bar=True,
                logger=True,
            )

    def validation_step(self, batch, batch_idx):
        x = batch["bag"]
        y = batch["label"]
        slide_name = batch["slide_name"]
        # patch_coords = batch["center_pixel"]
        loss, patch_preds, patch_scores = self.shared_step(x, y)
        slide_label = y[0]
        slide_score = patch_scores.mean(dim=0)  # of shape [num_classes]
        slide_pred = slide_score.argmax()

        self.val_acc(patch_preds, y)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "val/patch_acc",
            self.val_acc,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        return {
            "loss": loss,
            "patch_preds": patch_preds,
            "patch_scores": patch_scores,
            "patch_labels": y,
            "slide_score": slide_score,
            "slide_pred": slide_pred,
            "slide_name": slide_name,
            "slide_label": slide_label,
        }

    def validation_epoch_end(self, outputs):
        patch_scores = (
            torch.cat([x["patch_scores"] for x in outputs], dim=0).detach().cpu()
        )
        # patch_preds = (
        #     torch.cat([x["patch_preds"] for x in outputs], dim=0).detach().cpu()
        # )
        patch_labels = torch.cat([x["patch_labels"] for x in outputs], dim=0).cpu()
        slide_scores = torch.stack([x["slide_score"] for x in outputs]).detach().cpu()
        slide_preds = torch.stack([x["slide_pred"] for x in outputs]).detach().cpu()
        slide_names = [x["slide_name"] for x in outputs]
        slide_labels = torch.stack([x["slide_label"] for x in outputs]).cpu()

        self.log(
            "val/slide_acc",
            accuracy(
                slide_preds,
                slide_labels,
                task="multiclass",
                num_classes=self.num_classes,
            ),
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val/patch_auc",
            auroc(
                patch_scores,
                patch_labels,
                task="multiclass",
                num_classes=self.num_classes,
            ),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/slide_auc",
            auroc(
                slide_scores,
                slide_labels,
                task="multiclass",
                num_classes=self.num_classes,
            ),
            prog_bar=True,
            logger=True,
        )

        # TODO: gather all outputs to rank 0 and compute/log metrics there, this currently only logs for slides on rank 0
        if isinstance(self.logger, WandbLogger) and self.trainer.is_global_zero:
            self.logger.experiment.log(
                {"val/slide_roc": wandb.plot.roc_curve(slide_labels, slide_scores)},
            )

            # from this point on, we only support binary classification for now
            if self.num_classes > 2:
                return

            patch_scores, slide_scores = patch_scores[:, 1], slide_scores[:, 1]
            class_names = ["Negative", "Positive"]

            df = pd.DataFrame(
                data={
                    "slide_name": slide_names,
                    "score": slide_scores,
                    "target": slide_labels,
                }
            )

            # this also logs the table
            table = wandb.Table(data=df)
            self.logger.experiment.log(
                {
                    "val/slide_scores": wandb.plot.histogram(
                        table, "score", title="Val Slide Scores Histogram"
                    )
                }
            )

            cm = wandb.plot.confusion_matrix(
                y_true=slide_labels.numpy(),
                preds=slide_preds.numpy(),
                class_names=class_names,
            )
            self.logger.experiment.log({"val/conf_mat": cm})

    def test_step(self, batch, batch_idx):
        # TODO: patch/slide level testing
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        # TODO: patch/slide level testing
        raise NotImplementedError()

    def predict_step(self, batch, batch_idx):
        x = batch["patch"]
        features = self.forward_features(x)
        return features

    def shared_step(self, x, y):
        logits = self(x)
        loss = self.criterion(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        scores = logits.softmax(1)

        return loss, preds, scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=[p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.1
        )

        return [optimizer], [lr_scheduler]

    def _init_model(self, model, num_classes, ckpt_path, imagenet_pretrained, finetune):
        if model == "preact_resnet50":
            backbone = PreActResNet50()
            num_features = backbone.linear.in_features
            backbone.linear = nn.Identity()
        # elif model == "resnet50":
        #     # torchvision resnet
        #     backbone = resnet50(weights="DEFAULT" if imagenet_pretrained else None)
        #     num_features = backbone.fc.in_features
        #     backbone.fc = nn.Identity()
        else:
            # timm model
            backbone = timm.create_model(model, pretrained=imagenet_pretrained)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)

        classifier = nn.Linear(num_features, num_classes)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)

            # handle ssl pretrained model checkpoint
            for k in list(state_dict.keys()):
                if "backbone" in k:
                    state_dict[k.replace("backbone.", "")] = state_dict[k]
                del state_dict[k]

            incompatible_keys = backbone.load_state_dict(state_dict, strict=False)
            self.print(f"Loaded backbone from checkpoint at: {ckpt_path}")
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                self.print(
                    f"Incompatible keys when loading backbone from checkpoint: {incompatible_keys}"
                )

        if (imagenet_pretrained or ckpt_path is not None) and not finetune:
            for child in list(backbone.children()):
                for param in child.parameters():
                    param.requires_grad = False

        return backbone, classifier
