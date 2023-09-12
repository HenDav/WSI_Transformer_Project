from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import datasets


class WsiMilDataModule(LightningDataModule):
    name = "WsiMilDataModule"

    def __init__(
        self,
        dataset: str = "CAT",
        test_features_dir: str = "",
        bag_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 8,
        target: str = "ER",
        val_fold: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bag_size = bag_size
        # self.test_features_dir = test_features_dir if test_features_dir != '' else data_location[
        #     'TestSet Location']
        self.val_fold = val_fold

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_train = datasets.WSI_MILdataset(
                DataSet=self.dataset,
                bag_size=self.bag_size,
                target_kind=self.target,
                test_fold=self.val_fold,
                train=True,
                transform_type="pcbnfrsc",
            )

            self.dataset_val = datasets.WSI_MILdataset(
                DataSet=self.dataset,
                bag_size=self.bag_size,
                target_kind=self.target,
                test_fold=self.val_fold,
                train=False,
                transform_type="none",
            )
        elif stage == "test":
            self.dataset_test = datasets.WSI_MILdataset(
                DataSet=self.dataset,
                bag_size=self.bag_size,
                target_kind=self.target,
                test_fold=self.val_fold,
                train=False,
                transform_type="none",
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        # test_dset = datasets.Features_MILdataset(
        #     data_location=self.test_features_dir,
        #     dataset=self.dataset,
        #     target=self.target,
        #     is_all_tiles=True,
        #     minimum_tiles_in_slide=self.bag_size,
        #     is_train=False,
        #     test_fold=self.val_fold)

        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class WsiMilFeaturesDataModule(LightningDataModule):
    name = "WsiMilFeaturesDataModule"

    def __init__(
        self,
        dataset: str = "CAT",
        train_features_dir: str = "",
        data_location: dict = None,
        test_features_dir: str = "",
        test_dataset: str = "",
        bag_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 8,
        target: str = "ER",
        val_fold: int = 1,
        batch: int = 9,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # if it is of the form 'CARMEL->HAEMEK' make it 'HAEMEK' otherwise doesn't change it
        self.dataset = dataset.split('->')[-1] 
        self.data_location = data_location
        try:
            self.train_features_dir = (
                train_features_dir
                if train_features_dir != ""
                else data_location["TrainSet Location"]
            )
        except KeyError:
            self.train_features_dir = ""
            print(
                "Could not find train features dir data location in dict, ignore this if running inference only"
            )
        self.val_features_dir = data_location["TestSet Location"]
        self.test_features_dir = (
            test_features_dir
            if test_features_dir != ""
            else data_location["TestSet Location"]
        )
        if test_features_dir == "" and self.dataset == "CARMEL 9-11":
            self.test_features_dir = self.test_features_dir["Carmel " + str(batch)]
        self.test_dataset = test_dataset.split('->')[-1] if test_dataset != "" else self.dataset
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bag_size = bag_size
        self.val_fold = val_fold

    def setup(self, stage=None):
        if stage == "fit":
            if 'OR' not in self.target and 'with' not in self.target:
                self.dataset_train = datasets.Features_MILdataset(
                    dataset=self.dataset,
                    data_location=self.train_features_dir,
                    bag_size=self.bag_size,
                    target=self.target,
                    # minimum_tiles_in_slide=self.bag_size,
                    is_train=True,
                    test_fold=self.val_fold,
                )

                self.dataset_val = datasets.Features_MILdataset(
                    data_location=self.val_features_dir,
                    dataset=self.dataset,
                    bag_size=self.bag_size,
                    target=self.target,
                    minimum_tiles_in_slide=self.bag_size,
                    is_train=False,
                    test_fold=self.val_fold,
                )
            else:
                if 'with' in self.target: #meaning this is with cancer features
                    self.target = [self.target.split('_')[0]]*2
                self.dataset_train = datasets.Features_MILdataset_combined(
                    dataset=self.dataset,
                    data_location=self.train_features_dir,
                    bag_size=self.bag_size,
                    target=self.target,
                    # minimum_tiles_in_slide=self.bag_size,
                    is_train=True,
                    test_fold=self.val_fold,
                )

                self.dataset_val = datasets.Features_MILdataset_combined(
                    data_location=self.val_features_dir,
                    dataset=self.dataset,
                    bag_size=self.bag_size,
                    target=self.target,
                    minimum_tiles_in_slide=self.bag_size,
                    is_train=False,
                    test_fold=self.val_fold,
                )
        elif stage == "test":
            if 'OR' not in self.target and 'with' not in self.target:
                self.dataset_test = datasets.Features_MILdataset(
                    data_location=self.test_features_dir,
                    dataset=self.test_dataset,
                    target=self.target,
                    is_all_tiles=True,
                    minimum_tiles_in_slide=self.bag_size,
                    is_train=False,
                    test_fold=self.val_fold,
                )
            else:
                if 'with' in self.target: #meaning this is with cancer features
                    self.target = [self.target.split('_')[0]]*2
                self.dataset_test = datasets.Features_MILdataset_combined(
                    data_location=self.test_features_dir,
                    dataset=self.test_dataset,
                    target=self.target,
                    is_all_tiles=True,
                    minimum_tiles_in_slide=self.bag_size,
                    is_train=False,
                    test_fold=self.val_fold,
                )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
