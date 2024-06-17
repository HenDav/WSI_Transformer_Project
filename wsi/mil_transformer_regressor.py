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
from torch import nn
from torchmetrics.functional.regression import r2_score, mse, mean_absolute_error
from torchmetrics.functional.classification import auroc
from .models.metrics import c_index

from .models.loss import CoxPHLoss, DeepHitLoss, NaiveCensoredPinballLoss

from wsi.models.mil_transformer import MilTransformer
from wsi.wsi_regressor import WsiRegressor


class MilTransformerRegressor(LightningModule):
    def __init__(
        self,
        variant: Literal["vit", "simple"] = "vit",
        pos_encode: Literal["sincos", "learned", "None"] = "sincos",
        bag_size: int = 64,
        feature_dim: int = 512,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        loss: Literal["MSE", "Cox", "DeepHit", "NaiveCensoredPinballLoss"] = "Cox",
        survival = False,
        quantile_to_predict = 0.1,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
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
            dim: transformer dimension
            depth: transformer depth
            heads: number of attention heads
            mlp_dim: transformer mlp dimension
            dim_head: transformer dimension per head
            dropout: dropout rate
            emb_dropout: embedding dropout rate
            lr: learning rate
            feature_extractor_ckpt: path to pretrained WsiRegressor checkpoint for online feature extraction
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

        self.model = MilTransformer(
            variant=variant,
            pos_encode=pos_encode,
            bag_size=bag_size,
            input_dim=feature_dim,
            num_classes=1,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.loss = self.init_loss(loss, quantile_to_predict)

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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        loss, preds = self.shared_step(batch)

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

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs], dim=0).detach().cpu()
        loss = torch.cat([x["loss"].reshape(1) for x in outputs], dim=0).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()
            
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
                r2_score(preds, labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "train/MSE",
                mse(preds, labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "train/mean_absolute_error",
                mean_absolute_error(preds, labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        loss, preds = self.shared_step(batch)
        
        step_dict = {"loss": loss, "preds": preds, "labels": labels}

        if "secondary_label" in  batch.keys():
            secondary_label = batch["secondary_label"]
            step_dict["secondary_label"] = secondary_label

        return step_dict
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).detach().cpu()
        preds = torch.vstack([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.vstack([x["labels"] for x in outputs]).cpu()

        self.log(
            "val/loss",
            loss.mean(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.hparams.survival:
            pass
        else:
            self.log(
                "val/r2_score",
                r2_score(preds, labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "val/MSE",
                mse(preds, labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )
            self.log(
                "val/mean_absolute_error",
                mean_absolute_error(preds, labels),
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.hparams.batch_size,
                sync_dist=True,
            )

        df = pd.DataFrame(
            data={
                "prediction": preds.flatten(),
                "target": labels[:,0].flatten(),
            },
            
        )
        # if len(self.datasets_keys_val)>1:
        #     df["dataset_id"] = dataset_ids_series

        # this also logs the table
        table = wandb.Table(data=df)
        log_dict = {
                        "val/predictions": wandb.plot.histogram(
                            table, "prediction", title="Val Prediction Histogram"
                        ),
                    }
        self.logger.experiment.log(log_dict)
        

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        _, preds = self.shared_step(batch)

        pred = preds.mean(dim=0)
        label = labels[0]

        return {
            "pred": pred,
            "label": label,
            "slide_name": batch["slide_name"],
        }

    def test_epoch_end(self, outputs):
        slide_preds = torch.vstack([x["pred"] for x in outputs]).detach().cpu()
        slide_names = [x["slide_name"] for x in outputs]
        slide_labels = torch.vstack([x["label"] for x in outputs]).cpu()

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

        df_slides = pd.DataFrame(
            data={
                "slide_name": slide_names,
                "preds": slide_preds,
                "label": slide_labels,
            }
        )

        
        df_slides.to_csv(f"{self.logger.experiment.name}_preds.csv")

        if not isinstance(self.logger, WandbLogger):
            df_slides.to_csv(Path(self.logger.log_dir) / "slide_preds.csv")
            return

        # this also logs the table
        table = wandb.Table(data=df_slides.to_numpy(), columns=list(df_slides.columns))
        self.logger.experiment.log(
            {
                "test/slide_preds": wandb.plot.histogram(
                    table, "prediction", title="Test Slide Predictions Histogram"
                )
            }
        )

        df_slides.to_csv(Path(self.logger.experiment.dir) / "slide_preds.csv")


    def shared_step(self, batch):
        if self.use_features:
            x = batch["features"]
            y = batch["label"]
        else:
            bag = batch["bag"]
            y = batch["label"]
            with torch.no_grad():
                bag = rearrange(bag, "b1 b2 c h w -> (b1 b2) c h w")
                x = self.feature_extractor.forward_features(bag)
                x = rearrange(x, "(b1 b2) e -> b1 b2 e", b2=self.hparams.bag_size)

        preds = self.model(x)

        loss = self.loss(preds, y)

        return loss, preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
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
        feature_extractor = WsiRegressor.load_from_checkpoint(ckpt_path, model=model)
        feature_extractor.eval()

        features = feature_extractor.forward_features(torch.randn(1, 3, 256, 256))
        feature_dim = features.squeeze(0).shape[0]
        print(
            f"Initialized online feature extractor with {model} backbone, computed feature dim: {feature_dim}"
        )

        return feature_extractor, feature_dim
