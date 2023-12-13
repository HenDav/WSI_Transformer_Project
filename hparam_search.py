from pathlib import Path
import os

import torch
from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.cli import ArgsType, LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
# from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray import tune

from wsi.datasets.datamodules import WsiDataModule  # , LegacyWsiDataModule
from wsi.utils.features_writer import FeaturesWriter  # noqa: F401
from wsi.wsi_classifier import WsiClassifier


def train_wsi(config, num_epochs=100):
    seed_everything(1, workers=True)
    model = WsiClassifier(**config)
    dm = WsiDataModule(**config)
    metrics = {"patch_auc": "val/patch_auc", "slide_auc": "val/slide_auc"}

    logger = WandbLogger(project="hparam tuning WSI", entity="gipmed")
    trainer = Trainer(
        check_val_every_n_epoch=5,
        logger=logger,
        enable_checkpointing=False,
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_false",
        callbacks=[TuneReportCallback(metrics, on="validation_epoch_end")],
        enable_progress_bar = False)
    trainer.fit(model, dm)

if __name__ == "__main__":

    num_samples = 20
    num_epochs = 300
    gpus_per_trial = 1
    cpus_pre_run = 12

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-4),
        "lr_scheduler": tune.choice([True, False])
    }

    reporter = CLIReporter(
        parameter_columns=config.keys(),
        metric_columns=["patch_auc", "slide_auc", "training_iteration"])

    trainable = tune.with_parameters(
        train_wsi,
        num_epochs=num_epochs,)

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={
                "cpu": cpus_pre_run,
                "gpu": gpus_per_trial
            },
        ),
        tune_config=tune.TuneConfig(
            metric="slide_auc",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,)

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)