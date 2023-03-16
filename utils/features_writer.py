from pathlib import Path

import h5py
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter


class FeaturesWriter(BasePredictionWriter):
    """Callback to save extracted features from the model in h5 format"""

    def __init__(self, output_dir="./features"):
        super().__init__(write_interval="batch")

        self._slide_num = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            print(
                "WARNING: features output directory is not empty, features from slides with existing files will be appended"
            )
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
        slide_names_np = np.array(slide_names)
        slide_switch_indices = (
            np.where(slide_names_np[:-1] != slide_names_np[1:])[0] + 1
        )
        slide_switch_indices = np.insert(slide_switch_indices, 0, 0)
        slide_switch_indices = np.append(slide_switch_indices, [len(slide_names_np)])

        for i in range(len(slide_switch_indices) - 1):
            asset_dict = {
                "coords": coords[slide_switch_indices[i] : slide_switch_indices[i + 1]],
                "features": features[
                    slide_switch_indices[i] : slide_switch_indices[i + 1]
                ],
            }
            attr_dict = {
                "name": slide_names[slide_switch_indices[i]],
                "label": labels[slide_switch_indices[i]],
            }
            self._save_hdf5(slide_names[slide_switch_indices[i]], asset_dict, attr_dict)

    def _save_hdf5(self, slide_name, asset_dict, attr_dict=None):
        file_name = slide_name + "_features.h5"
        path = self.output_dir / file_name

        mode = "a" if path.exists() else "w"
        if mode == "w":  # new file
            print(f"Saving slide features in h5 file: {path}")
            self._slide_num += 1

        file = h5py.File(path, mode)
        if attr_dict is not None:
            metadata = file.require_group("metadata")
            for key, val in attr_dict.items():
                metadata.attrs[key] = val
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
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val
        file.close()
