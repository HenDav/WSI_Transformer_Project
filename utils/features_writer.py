from pathlib import Path

import h5py
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter


class FeaturesWriter(BasePredictionWriter):
    """Callback to save extracted features from the model in h5 format"""

    def __init__(self, output_dir="./features"):
        super().__init__(write_interval="batch")

        self._slide_num = 0
        self.output_dir = Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using FeaturesWriter, saving features to {self.output_dir}")

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        labels = batch["label"].cpu().numpy()
        slide_names = batch["slide_name"]
        coords = batch["center_pixel"].cpu().numpy()
        features = prediction.detach().cpu().numpy()

        # handle the case when the batch contains patches from more than one slide
        # assume it can contain at most two slides
        # check if the slide name changes in the batch
        slide_names_np = np.array(slide_names)
        slide_switch_indices = (
            np.where(slide_names_np[:-1] != slide_names_np[1:])[0] + 1
        )
        slide_switch_index = len(slide_names)
        if len(slide_switch_indices) == 1:
            slide_switch_index = slide_switch_indices[0]

        asset_dict = {
            "coords": coords[:slide_switch_index],
            "features": features[:slide_switch_index],
        }
        attr_dict = {"name": slide_names[0], "label": labels[0]}
        self._save_hdf5(asset_dict, attr_dict=attr_dict)

        if len(slide_switch_indices) == 1:
            asset_dict = {
                "coords": coords[slide_switch_index:],
                "features": features[slide_switch_index:],
            }
            attr_dict = {
                "name": slide_names[slide_switch_index],
                "label": labels[slide_switch_index],
            }
            self._save_hdf5(asset_dict, attr_dict=attr_dict)

    def _save_hdf5(self, asset_dict, attr_dict=None):
        path = self.output_dir / f"slide_{self._slide_num}.h5"
        mode = "a" if path.exists() else "w"
        if mode == "w":  # new file
            self._slide_num += 1

        file = h5py.File(path, mode)
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunk_shape,
                    dtype=data_type,
                )
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val
        file.close()
