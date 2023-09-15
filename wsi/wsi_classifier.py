from pathlib import Path
from typing import Literal, Optional, Dict

import pandas as pd
import numpy as np
import timm
import torch
from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR
import torchmetrics
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics.functional.classification import accuracy, auroc

from cleanlab.filter import find_label_issues

from .models.preact_resnet import PreActResNet50
from .core import constants


class WsiClassifier(LightningModule):
    def __init__(
        self,
        model: str = "resnet101",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        lr_scheduler: bool = False,
        num_classes: int = 2,
        ckpt_path: Optional[str] = None,
        imagenet_pretrained: bool = False,
        finetune: bool = False,
        criterion: Literal["crossentropy"] = "crossentropy",
        log_params: bool = False,
        batch_size: int = 256,
        train_classifier_from_scratch: bool = False,
        debug: bool = False,
        drop_rate: float = 0.0,
        datasets_folds: Dict = {},
        datasets_folds_val: Dict = {},
        log_scores: bool = False,
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

        datasets_folds = constants.get_datasets_folds(datasets_folds)
        datasets_folds_val = constants.get_datasets_folds(datasets_folds_val)

        self.datasets_keys = list(datasets_folds.keys())
        # self.datasets_keys_val = list(datasets_folds_val.keys())

        self.backbone, self.classifier = self._init_model(
            model, num_classes, ckpt_path, imagenet_pretrained, finetune, train_classifier_from_scratch, drop_rate
        )

        # TODO: support for more loss functions, balance weighting, etc
        self.criterion = nn.CrossEntropyLoss()

        self.num_classes = num_classes
        self.log_params = log_params
        
        self.debug = debug

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
        # if self.debug:
        #     slide_names = batch["slide_name"]
        #     center_pixels = batch["center_pixel"]
        loss, preds, scores = self.shared_step(x, y)

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
            accuracy(preds, y, task="multiclass", num_classes=self.num_classes),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        # if self.debug:
        #     return_tup = {"loss": loss, "patch_preds": preds, "patch_scores": scores, "y": y, "slide_names": slide_name, "center_pixels": center_pixel}
        # else:
        #     return_tup = {"loss": loss, "patch_preds": preds, "patch_scores": scores, "y": y}
        return {"loss": loss, "patch_preds": preds, "patch_scores": scores, "y": y}

    def training_epoch_end(self, outputs):
        scores = torch.cat([x["patch_scores"] for x in outputs], dim=0).detach().cpu()
        y = torch.cat([x["y"] for x in outputs], dim=0).cpu()
#         if self.debug: #TODO: fix
#             worst_examples = torch.cat([x["loss"].detach().cpu() for x in outputs], dim=0).sort(order="loss")[-100:]
#             best_examples = torch.cat([x[["loss", "slide_names"]] for x in outputs], dim=0).detach().cpu().sort(order="loss")[:100]
#             self.log(
#                 "train/worst_examples",
#                 worst_examples,
#                 logger=True
#             )
#             self.log(
#                 "train/best_examples",
#                 best_examples,
#                 logger=True
#             )
#             grid = torchvision.utils.make_grid(sample_imgs) 
#             self.logger.experiment.add_image('worst examples', grid, 0) 
            
        self.log(
            "train/patch_auc",
            auroc(scores, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        

    def validation_step(self, batch, batch_idx):
        x = batch["bag"]
        y = batch["label"]
        slide_name = batch["slide_name"]
        dataset_id = batch["dataset_id"]
        # patch_coords = batch["center_pixel"]
        loss, patch_preds, patch_scores = self.shared_step(x, y)
        slide_label = y[0]
        slide_score = patch_scores.mean(dim=0)  # of shape [num_classes]
        slide_pred = slide_score.argmax()
        
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
            accuracy(patch_preds, y, task="multiclass", num_classes=self.num_classes),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        step_dict = {
            "loss": loss,
            "patch_preds": patch_preds,
            "patch_scores": patch_scores,
            "patch_labels": y,
            "slide_score": slide_score,
            "slide_pred": slide_pred,
            "slide_name": slide_name,
            "dataset_id": dataset_id,
            "slide_label": slide_label,
        }

        if "secondary_label" in  batch.keys():
            secondary_patch_label = batch["secondary_label"]
            step_dict["secondary_patch_label"] = secondary_patch_label
            step_dict["secondary_slide_label"] = secondary_patch_label[0]

        return step_dict


    def validation_epoch_end(self, outputs_for_rank):
        outputs = self.all_gather(outputs_for_rank)
        patch_scores = (
            torch.cat([x["patch_scores"][i] for i in range(torch.distributed.get_world_size()) for x in outputs], dim=0).detach().cpu()
        )
        patch_labels = torch.cat([x["patch_labels"][i] for i in range(torch.distributed.get_world_size()) for x in outputs], dim=0).cpu()
        dataset_ids = np.asarray(torch.stack([x["dataset_id"][i] for i in range(torch.distributed.get_world_size()) for x in outputs], dim=0).cpu())
        slide_scores = torch.stack([x["slide_score"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).detach().cpu()
        slide_preds = torch.stack([x["slide_pred"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).detach().cpu()
        slide_labels = torch.stack([x["slide_label"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).cpu()
        
        if "secondary_patch_label" in outputs[0].keys():
            secondary_patch_labels = torch.cat([x["secondary_patch_label"][i] for i in range(torch.distributed.get_world_size()) for x in outputs], dim=0).cpu()
            secondary_slide_labels = torch.stack([x["secondary_slide_label"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).cpu()
            self.log(
                "val/secondary_patch_auc",
                auroc(patch_scores, secondary_patch_labels, task="multiclass", num_classes=self.num_classes),
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val/secondary_slide_auc",
                auroc(slide_scores, secondary_slide_labels, task="multiclass", num_classes=self.num_classes),
                prog_bar=True,
                logger=True,
            )
            
        # if len(self.datasets_keys_val)>1:
        #     dataset_ids_series = pd.Series(dataset_ids)
        #     for i in range(len(self.datasets_keys_val)):
        #         indices = np.argwhere(dataset_ids_series.array == i).flatten()
        #         if indices.size>3: # for the tracking not to trigger during the sanity check
        #             self.log(
        #                 f"{self.datasets_keys_val[i]}/slide_auc",
        #                 auroc(slide_scores[indices], slide_labels[indices], task="multiclass", num_classes=self.num_classes),
        #                 prog_bar=True,
        #                 logger=True,
        #             )
        #             if "secondary_patch_label" in outputs[0].keys():
        #                 self.log(
        #                     f"{self.datasets_keys_val[i]}/secondary_slide_auc",
        #                     auroc(slide_scores[indices], secondary_slide_labels[indices], task="multiclass", num_classes=self.num_classes),
        #                     prog_bar=True,
        #                     logger=True,
        #                 )
                
        #         if self.datasets_keys_val[i] == "TCGA_not_DX": # TODO:hard coded because of reasons :( can probably be elegantly fixed.
        #             slide_scores = np.delete(slide_scores, indices, axis=0)
        #             slide_labels = np.delete(slide_labels, indices, axis=0)
        #             slide_preds = np.delete(slide_preds, indices, axis=0)
        #             dataset_ids = np.delete(dataset_ids, indices, axis=0)
        #             dataset_ids_series = dataset_ids_series.drop(indices)
        #             if "secondary_patch_label" in outputs[0].keys():
        #                 secondary_slide_labels = np.delete(secondary_slide_labels, indices, axis=0)

        #             repetitions = outputs_for_rank[0]["patch_labels"].size()[0]
        #             expanded_indices = np.repeat(indices * repetitions, repetitions) + np.tile(np.arange(repetitions), len(indices))
        #             patch_labels = np.delete(patch_labels, expanded_indices, axis=0)
        #             patch_scores = np.delete(patch_scores, expanded_indices, axis=0)
        #         else:
        #             dataset_ids_series[indices] = self.datasets_keys_val[i]
        self.log(
            "val/slide_acc",
            accuracy(slide_preds, slide_labels, task="multiclass", num_classes=self.num_classes),
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val/patch_auc",
            auroc(patch_scores, patch_labels, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/slide_auc",
            auroc(slide_scores, slide_labels, task="multiclass", num_classes=self.num_classes),
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
                    "score": slide_scores,
                    "target": slide_labels,
                }
            )
            # if len(self.datasets_keys_val)>1:
            #     df["dataset_id"] = dataset_ids_series

            # this also logs the table
            table = wandb.Table(data=df)
            log_dict = {
                            "val/slide_scores": wandb.plot.histogram(
                                table, "score", title="Val Slide Scores Histogram"
                            ),
                        }
            if self.hparams.log_scores:
                log_dict[f"val/slide_scores_table_epoch_{self.current_epoch}"] = table
            self.logger.experiment.log(log_dict)

            cm = wandb.plot.confusion_matrix(
                y_true=slide_labels.numpy(),
                preds=slide_preds.numpy(),
                class_names=class_names,
            )
            self.logger.experiment.log({"val/conf_mat": cm})

    def test_step(self, batch, batch_idx):
        x = batch["bag"]
        y = batch["label"]
        slide_name = batch["slide_name"]
        dataset_id = batch["dataset_id"]
        center_pixels = batch["center_pixels"]
        # patch_coords = batch["center_pixel"]
        _, patch_preds, patch_scores = self.shared_step(x, y)
        slide_label = y[0]
        slide_score = patch_scores.mean(dim=0)  # of shape [num_classes]
        slide_pred = slide_score.argmax()

        return {
            "patch_preds": patch_preds,
            "patch_scores": patch_scores,
            "patch_labels": y,
            "slide_score": slide_score,
            "slide_pred": slide_pred,
            "slide_name": slide_name,
            "dataset_id": dataset_id,
            "slide_label": slide_label,
            "center_pixels": center_pixels,
        }

    def test_epoch_end(self, outputs):
        positive_patch_scores_per_slide = (
            torch.vstack([item['patch_scores'][:, 1].flatten() for item in outputs]).detach().cpu()
        )
        patch_scores = (
            torch.cat([item["patch_scores"] for item in outputs], dim=0).detach().cpu()
        )
        patch_labels = torch.cat([x["patch_labels"] for x in outputs], dim=0).cpu().to(int)
        slide_scores = torch.stack([x["slide_score"] for x in outputs]).detach().cpu()
        slide_preds = torch.stack([x["slide_pred"] for x in outputs]).detach().cpu()
        slide_names = [x["slide_name"] for x in outputs]
        # dataset_ids = [x["dataset_id"] for x in outputs]
        # center_pixels = torch.cat([x["center_pixels"] for x in outputs], dim=0).cpu()
        slide_labels = torch.stack([x["slide_label"] for x in outputs]).cpu()

        self.log(
            "test/slide_acc",
            accuracy(
                slide_preds,
                slide_labels,
                task="multiclass",
                num_classes=self.num_classes,
            ),
            logger=True,
        )

        self.log(
            "test/patch_auc",
            auroc(
                patch_scores,
                patch_labels,
                task="multiclass",
                num_classes=self.num_classes,
            ),
            logger=True,
        )
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

        # ranked_label_issues = find_label_issues(
        #     patch_labels.numpy(),
        #     patch_scores.numpy(),
        #     return_indices_ranked_by="self_confidence",
        # )

        # print(ranked_label_issues[:15])

        # print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
        # print(f"Top 15 most likely label errors: \n {pd.Series(slide_names)[ranked_label_issues][:15]}")
        # print(f"Top 15 most likely label errors coords: \n {pd.Series(center_pixels)[ranked_label_issues][:15]}")

        # from this point on, we only support binary classification for now
        if self.num_classes > 2:
            return

        positive_slide_scores = slide_scores[:, 1]
        class_names = ["Negative", "Positive"]
    
        df_patches = pd.DataFrame(positive_patch_scores_per_slide, columns=['patch_score_' + str(i) for i in range(positive_patch_scores_per_slide.shape[1])])

        df_slides = pd.DataFrame(
            data={
                "slide_name": slide_names,
                "score": positive_slide_scores,
                "label": slide_labels,
                # "dataset_ids": dataset_ids,
                # "center_pixels": center_pixels,
            }
        )

        df = pd.concat([df_slides, df_patches], axis=1)
        
        df.to_csv(f"{self.logger.experiment.name}_scores")

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
        table = wandb.Table(data=df.to_numpy(), columns=list(df.columns))
        self.logger.experiment.log(
            {
                "test/slide_scores": wandb.plot.histogram(
                    table, "score", title="Test Slide Scores Histogram"
                )

            }
        )

        cm = wandb.plot.confusion_matrix(
            y_true=slide_labels.numpy(),
            preds=slide_preds.numpy(),
            class_names=class_names,
        )
        self.logger.experiment.log({"test/conf_mat": cm})

        df.to_csv(Path(self.logger.experiment.dir) / "slide_scores.csv")

    def predict_step(self, batch, batch_idx):
        x = batch["patch"]
        features = self.forward_features(x)
        return features

    def shared_step(self, x, y):
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        scores = logits.softmax(1)
        if logits.isnan().any():
            print("some logits are none")

        return loss, preds, scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=0.25, total_iters=20),MultiStepLR(optimizer, milestones=[250,500], gamma=0.1)], milestones=[20])
        
        ret_value = {"optimizer":optimizer, "lr_scheduler":lr_scheduler, "monitor": "train/loss_epoch"} if self.hparams.lr_scheduler else [optimizer]

        return ret_value

    def _init_model(self, model, num_classes, ckpt_path, imagenet_pretrained, finetune, train_classifier_from_scratch, drop_rate):
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
            backbone = timm.create_model(model, pretrained=imagenet_pretrained, drop_rate=drop_rate)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)

        classifier = nn.Linear(num_features, num_classes)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)
            if "state_dict" in state_dict.keys(): #For ron's checkpoints
                state_dict = state_dict["state_dict"]
            
            # handle ssl pretrained model checkpoint
            for k in list(state_dict.keys()):
                if "backbone" in k:
                    state_dict[k.replace("backbone.", "")] = state_dict[k]
                    del state_dict[k]

            incompatible_keys_backbone = backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded backbone from checkpoint at: {ckpt_path}")
            if incompatible_keys_backbone.missing_keys or incompatible_keys_backbone.unexpected_keys:
                print(
                    f"Incompatible keys when loading backbone from checkpoint: {incompatible_keys_backbone}"
                )
            if not train_classifier_from_scratch:
                print("Loading classifier weights as well")
                for k in list(state_dict.keys()):
                    if "linear" in k:
                        state_dict[k.replace("linear.", "")] = state_dict[k]
                    elif "classifier" in k:
                        state_dict[k.replace("classifier.", "")] = state_dict[k]
                    del state_dict[k]
                incompatible_keys_classifier = classifier.load_state_dict(state_dict, strict=True)
                

        if (imagenet_pretrained or ckpt_path is not None) and not finetune:
            for child in list(backbone.children()):
                for param in child.parameters():
                    param.requires_grad = False

        return backbone, classifier
