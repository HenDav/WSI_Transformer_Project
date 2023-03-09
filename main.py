from pathlib import Path

import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import ArgsType, LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from datasets.datamodules import WsiDataModule
from utils.features_writer import FeaturesWriter  # noqa: F401
from wsi_classifier import WsiClassifier


class WsiLightningCLI(LightningCLI):
    def before_fit(self):
        # allow specifying wandb checkpoint paths in the form of "wandb:USER/PROJECT/MODEL-RUN_ID:VERSION"
        # reference can be retrieved in artifacts panel
        # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best_k")

        ckpt_path = vars(self.config).get("ckpt_path")

        if ckpt_path is not None and ckpt_path.startswith("wandb:"):
            wandb_reference = ckpt_path.split(":")[1]
            # download wandb checkpoint and update ckpt_path in args
            artifact_dir = WandbLogger.download_artifact(
                wandb_reference, artifact_type="model"
            )
            self.config["ckpt_path"] = Path(artifact_dir) / "model.ckpt"


def cli_main(args: ArgsType = None):
    lr_monitor = LearningRateMonitor()
    trainer_defaults = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto",
        "max_epochs": 100,
        "callbacks": [lr_monitor],
    }

    # note the current run's generated config.yaml file is saved in the cwd and not logged to wandb atm, it is overwritten every run
    # follow https://github.com/Lightning-AI/lightning/issues/14188 for the fix
    cli = WsiLightningCLI(  # noqa: F841
        WsiClassifier,
        WsiDataModule,
        trainer_defaults=trainer_defaults,
        seed_everything_default=True,
        parser_kwargs={"fit": {"default_config_files": ["default_config_fit.yaml"]}},
        save_config_kwargs={"overwrite": True},
        args=args,
    )


if __name__ == "__main__":
    cli_main()
