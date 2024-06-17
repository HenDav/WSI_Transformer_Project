# Transformer sampling strategies for Whole Slide Images

This project is an implementation of our final project for Transformers - 236004@Technion.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To install this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/HenDav/WSI_Transformer_Project
    cd WSI_Transformer_Project
    ```
2. Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate conda_master
    ```

## Usage

To use this project, follow these steps:

1. Run the main script for training the model:
    ```bash
    python main_mil_transformer.py fit --model.variant simple --trainer.logger.init_args.name <wandb run name> --trainer.logger.init_args.project <wandb project name> --data.features_dir <patch features dir> --model.feature_dim <feature dimention> --data.test_features_dir <test patch features dir> --trainer.max_epochs 20 --data.bag_size 100 --patch_sampling <patch sampling stratagy> --model.pos_encode <positional encoding> (if model.pos_encode==multi_grid, you'll also need to include --data.num_grids 25 --model.num_grids_pos_encode 25)
    ```
2. Run the test:
    ```bash
    python main_mil_transformer.py test --ckpt_path <our checkpoint for the model trained> --data.datasets_folds "{TCGA: [6]}" --model.variant simple --trainer.logger.init_args.name <wandb run name> --trainer.logger.init_args.project <wandb project name> --data.features_dir <patch features dir> --model.feature_dim <feature dimention> --data.test_features_dir <test patch features dir> --trainer.max_epochs 20 --data.bag_size 100 --patch_sampling <patch sampling stratagy> --model.pos_encode <positional encoding> (if model.pos_encode==multi_grid, you'll also need to include --data.num_grids 25 --model.num_grids_pos_encode 25)
    ```

For the features used in training and testing contact dahen@cs.technion.ac.il