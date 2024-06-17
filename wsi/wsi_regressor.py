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
from torchmetrics.functional.regression import r2_score, mse, mean_absolute_error
from torchmetrics.functional.classification import auroc
from .models.metrics import c_index

from .models.loss import CoxPHLoss, DeepHitLoss, NaiveCensoredPinballLoss
from pycox.evaluation.eval_surv import EvalSurv # Use later

import sys
sys.path.append("../wsi-ssl")
import wsi_ssl # circular import here for ssl model loading. needs to be taken care of at some point.
from .models.preact_resnet import PreActResNet50
from .core import constants


class WsiRegressor(LightningModule):
    """
    WsiRegressor is a PyTorch Lightning module for Whole Slide Image regression tasks. 
    The regressor uses a specified model architecture and applies specific learning rate, 
    weight decay, and other training configurations. It supports multi-class classification 
    and allows for fine-tuning and checkpoint loading.

    Attributes:
        backbone: The backbone of the model.
        regressor: The regressor part of the model.
        criterion: The criterion or loss function used by the model.    
        num_classes: The number of classes for classification.
        log_params: A flag indicating whether to log parameters.
        debug: A flag indicating whether the model is in debug mode.    

    Example:
        model = WsiRegressor(model="resnet101", num_classes=3)
        trainer = Trainer(max_epochs=10)
        trainer.fit(model, train_dataloader, val_dataloader)
    """
    def __init__(
        self,
        model: str = "resnet101",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        lr_scheduler: bool = False,
        ckpt_path: Optional[str] = None,
        imagenet_pretrained: bool = False,
        finetune: bool = False,
        loss: Literal["MSE", "Cox", "DeepHit", "NaiveCensoredPinballLoss"] = "Cox",
        log_params: bool = False,
        batch_size: int = 256,
        train_regressor_from_scratch: bool = False,
        debug: bool = False,
        drop_rate: float = 0.0,
        survival = False,
        quantile_to_predict = 0.1,
        log_preds = False,
        **kwargs,
    ):
        """
        Initializes the WsiRegressor object.

        Args:
            model (str, optional): Backbone model, either a timm model or "preact_resnet50". Defaults to "resnet101".
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
            lr_scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to False.
            ckpt_path (Optional[str], optional): Path to pretrained backbone checkpoint. If not provided, defaults to None.
            imagenet_pretrained (bool, optional): Whether to use ImageNet pretrained weights for the backbone. Defaults to False.
            finetune (bool, optional): Whether to finetune the backbone. Defaults to False.
            criterion (Literal["crossentropy"], optional): Loss function to use. Defaults to "crossentropy".
            log_params (bool, optional): Whether to log model parameters and gradients to wandb. Defaults to False.
            batch_size (int, optional): Batch size for data loading. Defaults to 256.
            train_regressor_from_scratch (bool, optional): Whether to train the regressor from scratch. Defaults to False.
            debug (bool, optional): Whether to run in debug mode. Defaults to False.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.save_hyperparameters()

        self.backbone, self.regressor = self._init_model(
            model, ckpt_path, imagenet_pretrained, finetune, train_regressor_from_scratch, drop_rate
        )

        self.loss = self.init_loss(loss, quantile_to_predict)

        self.log_params = log_params
        
        self.debug = debug

    def init_loss(self, loss, quantile_to_predict):
        if loss == "MSE":
            _loss = nn.MSELoss()
        elif loss == "Cox":
            _loss = CoxPHLoss()
        elif loss == "DeepHit":
            _loss = DeepHitLoss()
        elif loss == "NaiveCensoredPinballLoss":
            _loss = NaiveCensoredPinballLoss(alpha=quantile_to_predict)
        return _loss

    def forward(self, x):
        return self.regressor(self.backbone(x))

    def forward_features(self, x):
        return self.backbone(x)

    def on_fit_start(self):
        if self.log_params:
            self.logger.watch(self, log="all")

    def training_step(self, batch, batch_idx):
        x = batch["patch"]
        y = batch["label"]
        loss, preds = self.shared_step(x, y)

        self.log(
            "train/step_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        return {"loss": loss, "patch_preds": preds, "y": y}

    def training_epoch_end(self, outputs):
        preds = torch.cat([x["patch_preds"] for x in outputs], dim=0).detach().cpu()
        loss = torch.cat([x["loss"].reshape(1) for x in outputs], dim=0).detach().cpu()
        y = torch.cat([x["y"] for x in outputs], dim=0).cpu()
            
        self.log(
            "train/loss",
            loss.mean(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.hparams.survival:
            pass
        else:
            self.log(
                "train/r2_score",
                r2_score(preds, y),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "train/MSE",
                mse(preds, y),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "train/mean_absolute_error",
                mean_absolute_error(preds, y),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
        

    def validation_step(self, batch, batch_idx):
        x = batch["bag"]
        y = batch["label"]
        slide_name = batch["slide_name"]
        dataset_id = batch["dataset_id"]
        patch_loss, patch_preds = self.shared_step(x, y)
        slide_label = y[0]
        slide_pred = patch_preds.mean()  # of shape [num_classes]
        
        self.log(
            "val/patch_loss",
            patch_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        step_dict = {
            "patch_loss": patch_loss,
            "patch_preds": patch_preds,
            "patch_labels": y,
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
        patch_labels = torch.cat([x["patch_labels"][i] for i in range(torch.distributed.get_world_size()) for x in outputs], dim=0).cpu()
        slide_preds = torch.stack([x["slide_pred"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).detach().cpu()
        slide_labels = torch.stack([x["slide_label"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).cpu()
        patch_loss = torch.cat([x["patch_loss"].reshape(1) for x in outputs], dim=0).detach().cpu()
        
        self.log(
            "val/patch_loss",
            patch_loss.mean(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )
        if "secondary_patch_label" in outputs[0].keys():
            secondary_patch_labels = torch.cat([x["secondary_patch_label"][i] for i in range(torch.distributed.get_world_size()) for x in outputs], dim=0).cpu()
            secondary_slide_labels = torch.stack([x["secondary_slide_label"][i] for i in range(torch.distributed.get_world_size()) for x in outputs]).cpu()
            self.log(
                "val/secondary_patch_auc",
                auroc(slide_preds, secondary_patch_labels, task="binary"),
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val/secondary_slide_auc",
                auroc(slide_preds, secondary_slide_labels, task="binary"),
                prog_bar=True,
                logger=True,
            )

        if self.hparams.survival:
            # self.log(
            #     "val/c_index",
            #     c_index(
            #         slide_preds,
            #         slide_labels
            #     ),
            #     logger=True,
            # )
            pass
        else:
            self.log(
                "val/r2_score",
                r2_score(slide_preds, slide_labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "val/MSE",
                mse(slide_preds, slide_labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "val/mean_absolute_error",
                mean_absolute_error(slide_preds, slide_labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )

        # TODO: gather all outputs to rank 0 and compute/log metrics there, this currently only logs for slides on rank 0
        if isinstance(self.logger, WandbLogger) and self.trainer.is_global_zero:
            if self.hparams.survival:
                slide_labels = slide_labels.T[0]
            df = pd.DataFrame(
                data={
                    "preds": slide_preds,
                    "target": slide_labels,
                }
            )
            # if len(self.datasets_keys_val)>1:
            #     df["dataset_id"] = dataset_ids_series

            # this also logs the table
            table = wandb.Table(data=df)
            log_dict = {
                            "val/slide_preds": wandb.plot.histogram(
                                table, "preds", title="Val Slide Predictions Histogram"
                            ),
                        }
            if self.hparams.log_preds:
                log_dict[f"val/slide_preds_table_epoch_{self.current_epoch}"] = table
            self.logger.experiment.log(log_dict)

    def test_step(self, batch, batch_idx):
        x = batch["bag"]
        label = batch["label"]
        slide_name = batch["slide_name"]
        dataset_id = batch["dataset_id"]
        center_pixels = batch["center_pixels"]
        _, patch_preds = self.shared_step(x, y)
        slide_label = y[0]
        slide_pred = patch_preds.mean()

        return {
            "patch_preds": patch_preds,
            "patch_labels": label,
            "slide_pred": slide_pred,
            "slide_name": slide_name,
            "dataset_id": dataset_id,
            "slide_label": slide_label,
            "center_pixels": center_pixels,
        }

    def test_epoch_end(self, outputs):
        patch_preds_per_slide = (
            torch.vstack([item['patch_preds'].flatten() for item in outputs]).detach().cpu()
        )
        patch_preds = (
            torch.cat([item["patch_preds"] for item in outputs], dim=0).detach().cpu()
        )
        patch_labels = torch.cat([x["patch_labels"] for x in outputs], dim=0).cpu().to(int)
        slide_preds = torch.stack([x["slide_pred"] for x in outputs]).detach().cpu()
        slide_names = [x["slide_name"] for x in outputs]
        slide_labels = torch.stack([x["slide_label"] for x in outputs]).cpu()

        if self.hparams.survival:
            self.log(
                "test/c_index",
                c_index(
                    slide_preds,
                    slide_labels
                ),
                logger=True,
            )

        else:
            self.log(
                "val/r2_score",
                r2_score(slide_preds, slide_labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "val/MSE",
                mse(slide_preds, slide_labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "val/mean_absolute_error",
                mean_absolute_error(slide_preds, slide_labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )

        df_patches = pd.DataFrame(patch_preds_per_slide, columns=['patch_pred_' + str(i) for i in range(patch_preds_per_slide.shape[1])])

        df_slides = pd.DataFrame(
            data={
                "slide_name": slide_names,
                "preds": slide_preds,
                "label": slide_labels,
            }
        )

        df = pd.concat([df_slides, df_patches], axis=1)
        
        df.to_csv(f"{self.logger.experiment.name}_preds.csv")

        if not isinstance(self.logger, WandbLogger):
            df.to_csv(Path(self.logger.log_dir) / "slide_preds.csv")
            return

        # this also logs the table
        table = wandb.Table(data=df.to_numpy(), columns=list(df.columns))
        self.logger.experiment.log(
            {
                "test/slide_preds": wandb.plot.histogram(
                    table, "prediction", title="Test Slide Predictions Histogram"
                )
            }
        )

        df.to_csv(Path(self.logger.experiment.dir) / "slide_preds.csv")

    def predict_step(self, batch, batch_idx):
        x = batch["patch"]
        features = self.forward_features(x)
        return features

    def shared_step(self, x, y):
        preds = self(x)
        loss = self.loss(preds, y)
        if preds.isnan().any():
            print("some preds are none")

        return loss, preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=0.25, total_iters=20),MultiStepLR(optimizer, milestones=[250,500], gamma=0.1)], milestones=[20])
        
        ret_value = {"optimizer":optimizer, "lr_scheduler":lr_scheduler, "monitor": "train/loss_epoch"} if self.hparams.lr_scheduler else [optimizer]

        return ret_value

    def _init_model(self, model, ckpt_path, imagenet_pretrained, finetune, train_regressor_from_scratch, drop_rate):
        if model == "preact_resnet50":
            backbone = PreActResNet50()
            num_features = backbone.linear.in_features
            backbone.linear = nn.Identity()
        else:
            # timm model
            backbone = timm.create_model(model, pretrained=imagenet_pretrained, drop_rate=drop_rate)
            num_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0)

        regressor = nn.Linear(num_features, 1)

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
            if not train_regressor_from_scratch:
                print("Loading regressor weights as well")
                for k in list(state_dict.keys()):
                    if "linear" in k:
                        state_dict[k.replace("linear.", "")] = state_dict[k]
                    elif "regressor" in k:
                        state_dict[k.replace("regressor.", "")] = state_dict[k]
                    del state_dict[k]
                incompatible_keys_regressor = regressor.load_state_dict(state_dict, strict=False)
                if incompatible_keys_regressor.missing_keys or incompatible_keys_regressor.unexpected_keys:
                    print(
                        f"Incompatible keys when loading regressor from checkpoint: {incompatible_keys_regressor}"
                    )

        if (imagenet_pretrained or ckpt_path is not None) and not finetune:
            for child in list(backbone.children()):
                for param in child.parameters():
                    param.requires_grad = False

        return backbone, regressor
