from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
import torchmetrics
import wandb
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional import auroc

from models.mil_transformer import MilTransformer
from wsi.wsi_classifier import WsiClassifier


class MilTransformerClassifier(LightningModule):
    def __init__(
        self,
        variant: Literal["vit", "simple"] = "vit",
        pos_encode: Literal["sincos", "learned"] = "sincos",
        bag_size: int = 64,
        feature_dim: int = 512,
        num_classes: int = 2,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        lr: float = 1e-3,
        feature_extractor_ckpt: Optional[str] = None,
        feature_extractor_backbone: Optional[str] = None,
        batch_size: int = 128,
        **kwargs,
    ):
        """
        Args:
            variant: transformer variant
            pos_encode: positional encoding type
            bag_size: number of patches per bag,
            feature_dim: patch embedding input dimension
            num_classes: number of target classes
            dim: transformer dimension
            depth: transformer depth
            heads: number of attention heads
            mlp_dim: transformer mlp dimension
            dim_head: transformer dimension per head
            dropout: dropout rate
            emb_dropout: embedding dropout rate
            lr: learning rate
            feature_extractor_ckpt: path to pretrained WsiClassifier checkpoint for online feature extraction
            feature_extractor_backbone: name of feature extractor backbone in feature extractor checkpoint
            batch_size: batch size
        """
        super().__init__()

        self.save_hyperparameters()

        self.feature_extractor = None
        self.use_features = True
        if feature_extractor_ckpt is not None:
            assert (
                feature_extractor_backbone is not None
            ), "must provide name of feature extractor backbone when doing online extraction"
            self.feature_extractor, feature_dim = self._init_feature_extractor(
                feature_extractor_backbone, feature_extractor_ckpt
            )
            self.use_features = False
        self.num_classes = num_classes

        self.model = MilTransformer(
            variant=variant,
            pos_encode=pos_encode,
            bag_size=bag_size,
            input_dim=feature_dim,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        loss, preds, scores, y = self.shared_step(batch)
        self.train_acc(preds, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return {"loss": loss, "scores": scores, "y": y}

    def training_epoch_end(self, outputs):
        scores = torch.cat([x["scores"] for x in outputs], dim=0).detach().cpu()
        y = torch.cat([x["y"] for x in outputs], dim=0).cpu()

        if self.num_classes == 2:
            self.log(
                "train/slide_auc",
                auroc(scores[:, 1], y, task="binary"),
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "train/slide_auc",
                auroc(scores, y, task="multiclass"),
                prog_bar=True,
                logger=True,
            )

    def validation_step(self, batch, batch_idx):
        loss, preds, scores, y = self.shared_step(batch)
        self.val_acc(scores, y)

        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "val/bag_acc",
            self.val_acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

        return {"loss": loss, "scores": scores, "y": y}

    def validation_epoch_end(self, outputs):
        scores = torch.cat([x["scores"] for x in outputs], dim=0).detach().cpu()
        y = torch.cat([x["y"] for x in outputs], dim=0).cpu()

        if self.num_classes == 2:
            self.log(
                "val/slide_auc",
                auroc(scores[:, 1], y, task="binary"),
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "val/slide_auc",
                auroc(scores, y, task="multiclass"),
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        _, _, scores, y = self.shared_step(batch)

        slide_score = scores.mean(dim=0)  # shape (num_classes)
        slide_label = y[0]

        return {
            "score": slide_score,
            "label": slide_label,
            "slide_name": batch["slide_name"],
        }

    def test_epoch_end(self, outputs):
        slide_scores = torch.stack([x["score"] for x in outputs]).detach().cpu()
        slide_names = [x["slide_name"] for x in outputs]
        slide_labels = torch.stack([x["label"] for x in outputs]).cpu()

        self.log(
            "test/slide_auc",
            auroc(
                slide_scores,
                slide_labels,
                task="multiclass",
                num_classes=self.num_classes,
            ),
            logger=True,
        )

        # from this point on, we only support binary classification for now
        if self.num_classes > 2:
            return

        positive_slide_scores = slide_scores[:, 1]
        class_names = ["Negative", "Positive"]

        df = pd.DataFrame(
            data={
                "slide_name": slide_names,
                "score": positive_slide_scores,
                "label": slide_labels,
            }
        )

        if not isinstance(self.logger, WandbLogger):
            df.to_csv(Path(self.logger.log_dir) / "slide_scores.csv")
            return

        self.logger.experiment.log(
            {
                "test/slide_roc": wandb.plot.roc_curve(
                    slide_labels.unsqueeze(1), slide_scores, labels=class_names
                )
            },
        )

        # this also logs the table
        table = wandb.Table(data=df)
        self.logger.experiment.log(
            {
                "test/slide_scores": wandb.plot.histogram(
                    table, "score", title="Test Slide Scores Histogram"
                )
            }
        )

        df.to_csv(Path(self.logger.experiment.dir) / "slide_scores.csv")

    def shared_step(self, batch):
        if self.use_features:
            x = batch["features"]
            y = batch["label"]
            # patch_coords = batch["coords"]
        else:
            bag = batch["bag"]
            y = batch["label"]
            # patch_coords = batch["center_pixel"]
            with torch.no_grad():
                bag = rearrange(bag, "b1 b2 c h w -> (b1 b2) c h w")
                x = self.feature_extractor.forward_features(bag)
                x = rearrange(x, "(b1 b2) e -> b1 b2 e", b2=self.hparams.bag_size)

        y = y[:, 0]
        logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        scores = logits.softmax(1)
        loss = F.cross_entropy(logits, y)

        return loss, preds, scores, y

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(0.5 * self.trainer.max_epochs),
                    int(0.7 * self.trainer.max_epochs),
                ],
                gamma=0.1,
            ),
            "interval": "epoch",
        }
        return [optimizer], [lr_scheduler]

    def _init_feature_extractor(self, model, ckpt_path):
        feature_extractor = WsiClassifier.load_from_checkpoint(ckpt_path, model=model)
        feature_extractor.eval()

        features = feature_extractor.forward_features(torch.randn(1, 3, 256, 256))
        feature_dim = features.squeeze(0).shape[0]
        self.print(
            f"Initialized online feature extractor with {model} backbone, computed feature dim: {feature_dim}"
        )

        return feature_extractor, feature_dim
