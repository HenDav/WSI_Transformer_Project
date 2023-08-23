from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy
import pandas

from core import constants, utils
from core.base import OutputObject


# =================================================
# MetadataGenerator Class
# =================================================
class MetadataBase(ABC):
    def __init__(
        self,
        datasets_base_dir_path: Path,
        tile_size: int,
        metadata_at_magnification: int,
        **kw: object,
    ):
        super().__init__()
        self._datasets_base_dir_path = datasets_base_dir_path
        self._tile_size = tile_size
        self._metadata_at_magnification = metadata_at_magnification
        self._dataset_paths = constants.get_dataset_paths(
            datasets_base_dir_path=datasets_base_dir_path
        )
        self._df = self._load_metadata()

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    @abstractmethod
    def _load_metadata(self) -> pandas.DataFrame:
        pass


# =================================================
# MetadataGenerator Class
# =================================================
class MetadataGenerator(OutputObject, MetadataBase):
    def __init__(
        self,
        name: str,
        output_dir_path: Path,
        datasets_base_dir_path: Path,
        tile_size: int,
        desired_magnification: int,
        metadata_enhancement_dir_path: Path,
        dataset_ids: List[str],
    ):
        self._metadata_enhancement_dir_path = metadata_enhancement_dir_path
        self._dataset_ids = dataset_ids
        super().__init__(
            name=name,
            output_dir_path=output_dir_path,
            datasets_base_dir_path=datasets_base_dir_path,
            tile_size=tile_size,
            metadata_at_magnification=desired_magnification,
        )

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    def save_metadata(self):
        self._output_dir_path.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(path_or_buf=self._output_dir_path / "metadata.csv")

    def _build_column_names(self):
        column_names = {}
        for dataset_id_prefix in constants.metadata_base_dataset_ids:
            column_names[dataset_id_prefix] = {}
            column_names[dataset_id_prefix][
                self._get_total_tiles_column_name()
            ] = constants.total_tiles_column_name
            column_names[dataset_id_prefix][
                self._get_legitimate_tiles_column_name()
            ] = constants.legitimate_tiles_column_name
            column_names[dataset_id_prefix][
                self._get_slide_tile_usage_column_name(
                    dataset_id_prefix=dataset_id_prefix
                )
            ] = constants.tile_usage_column_name

            if dataset_id_prefix.startswith(constants.dataset_id_abctb):
                column_names[dataset_id_prefix][
                    constants.file_column_name_abctb
                ] = constants.file_column_name
                column_names[dataset_id_prefix][
                    constants.patient_barcode_column_name_abctb
                ] = constants.patient_barcode_column_name
                column_names[dataset_id_prefix][
                    constants.dataset_id_column_name_abctb
                ] = constants.dataset_id_column_name
                column_names[dataset_id_prefix][
                    constants.mpp_column_name_abctb
                ] = constants.mpp_column_name
                column_names[dataset_id_prefix][
                    constants.scan_date_column_name_abctb
                ] = constants.scan_date_column_name
                column_names[dataset_id_prefix][
                    constants.width_column_name_abctb
                ] = constants.width_column_name
                column_names[dataset_id_prefix][
                    constants.height_column_name_abctb
                ] = constants.height_column_name
                column_names[dataset_id_prefix][
                    constants.magnification_column_name_abctb
                ] = constants.magnification_column_name
                column_names[dataset_id_prefix][
                    constants.er_status_column_name_abctb
                ] = constants.er_status_column_name
                column_names[dataset_id_prefix][
                    constants.pr_status_column_name_abctb
                ] = constants.pr_status_column_name
                column_names[dataset_id_prefix][
                    constants.her2_status_column_name_abctb
                ] = constants.her2_status_column_name
                column_names[dataset_id_prefix][
                    constants.fold_column_name_abctb
                ] = constants.fold_column_name

            if dataset_id_prefix.startswith(constants.dataset_id_tcga):
                column_names[dataset_id_prefix][
                    constants.file_column_name_tcga
                ] = constants.file_column_name
                column_names[dataset_id_prefix][
                    constants.patient_barcode_column_name_tcga
                ] = constants.patient_barcode_column_name
                column_names[dataset_id_prefix][
                    constants.dataset_id_column_name_tcga
                ] = constants.dataset_id_column_name
                column_names[dataset_id_prefix][
                    constants.mpp_column_name_tcga
                ] = constants.mpp_column_name
                column_names[dataset_id_prefix][
                    constants.scan_date_column_name_tcga
                ] = constants.scan_date_column_name
                column_names[dataset_id_prefix][
                    constants.width_column_name_tcga
                ] = constants.width_column_name
                column_names[dataset_id_prefix][
                    constants.height_column_name_tcga
                ] = constants.height_column_name
                column_names[dataset_id_prefix][
                    constants.magnification_column_name_tcga
                ] = constants.magnification_column_name
                column_names[dataset_id_prefix][
                    constants.er_status_column_name_tcga
                ] = constants.er_status_column_name
                column_names[dataset_id_prefix][
                    constants.pr_status_column_name_tcga
                ] = constants.pr_status_column_name
                column_names[dataset_id_prefix][
                    constants.her2_status_column_name_tcga
                ] = constants.her2_status_column_name
                column_names[dataset_id_prefix][
                    constants.fold_column_name_tcga
                ] = constants.fold_column_name

            if dataset_id_prefix.startswith(constants.dataset_id_haemek):
                column_names[dataset_id_prefix][
                    constants.file_column_name_haemek
                ] = constants.file_column_name
                column_names[dataset_id_prefix][
                    constants.patient_barcode_column_name_haemek
                ] = constants.patient_barcode_column_name
                column_names[dataset_id_prefix][
                    constants.dataset_id_column_name_haemek
                ] = constants.dataset_id_column_name
                column_names[dataset_id_prefix][
                    constants.mpp_column_name_haemek
                ] = constants.mpp_column_name
                column_names[dataset_id_prefix][
                    constants.scan_date_column_name_haemek
                ] = constants.scan_date_column_name
                column_names[dataset_id_prefix][
                    constants.width_column_name_haemek
                ] = constants.width_column_name
                column_names[dataset_id_prefix][
                    constants.height_column_name_haemek
                ] = constants.height_column_name
                column_names[dataset_id_prefix][
                    constants.magnification_column_name_haemek
                ] = constants.magnification_column_name
                column_names[dataset_id_prefix][
                    constants.er_status_column_name_haemek
                ] = constants.er_status_column_name
                column_names[dataset_id_prefix][
                    constants.pr_status_column_name_haemek
                ] = constants.pr_status_column_name
                column_names[dataset_id_prefix][
                    constants.ki_67_status_column_name_haemek
                ] = constants.ki_67_status_column_name
                column_names[dataset_id_prefix][
                    constants.her2_status_column_name_haemek
                ] = constants.her2_status_column_name
                column_names[dataset_id_prefix][
                    constants.fold_column_name_haemek
                ] = constants.fold_column_name
                column_names[dataset_id_prefix][
                    constants.tumor_type_column_name_haemek
                ] = constants.tumor_type_column_name

            if dataset_id_prefix.startswith(constants.dataset_id_carmel):
                column_names[dataset_id_prefix][
                    constants.file_column_name_carmel
                ] = constants.file_column_name
                column_names[dataset_id_prefix][
                    constants.patient_barcode_column_name_carmel
                ] = constants.patient_barcode_column_name
                column_names[dataset_id_prefix][
                    constants.dataset_id_column_name_carmel
                ] = constants.dataset_id_column_name
                column_names[dataset_id_prefix][
                    constants.mpp_column_name_carmel
                ] = constants.mpp_column_name
                column_names[dataset_id_prefix][
                    constants.scan_date_column_name_carmel
                ] = constants.scan_date_column_name
                column_names[dataset_id_prefix][
                    constants.width_column_name_carmel
                ] = constants.width_column_name
                column_names[dataset_id_prefix][
                    constants.height_column_name_carmel
                ] = constants.height_column_name
                column_names[dataset_id_prefix][
                    constants.magnification_column_name_carmel
                ] = constants.magnification_column_name
                column_names[dataset_id_prefix][
                    constants.er_status_column_name_carmel
                ] = constants.er_status_column_name
                column_names[dataset_id_prefix][
                    constants.pr_status_column_name_carmel
                ] = constants.pr_status_column_name
                column_names[dataset_id_prefix][
                    constants.ki_67_status_column_name_carmel
                ] = constants.ki_67_status_column_name
                column_names[dataset_id_prefix][
                    constants.her2_status_column_name_carmel
                ] = constants.her2_status_column_name
                column_names[dataset_id_prefix][
                    constants.fold_column_name_carmel
                ] = constants.fold_column_name

            if dataset_id_prefix.startswith(constants.dataset_id_sheba):
                column_names[dataset_id_prefix][
                    constants.file_column_name_sheba
                ] = constants.file_column_name
                column_names[dataset_id_prefix][
                    constants.patient_barcode_column_name_sheba
                ] = constants.patient_barcode_column_name
                column_names[dataset_id_prefix][
                    constants.dataset_id_column_name_sheba
                ] = constants.dataset_id_column_name
                column_names[dataset_id_prefix][
                    constants.mpp_column_name_sheba
                ] = constants.mpp_column_name
                column_names[dataset_id_prefix][
                    constants.scan_date_column_name_sheba
                ] = constants.scan_date_column_name
                column_names[dataset_id_prefix][
                    constants.width_column_name_sheba
                ] = constants.width_column_name
                column_names[dataset_id_prefix][
                    constants.height_column_name_sheba
                ] = constants.height_column_name
                column_names[dataset_id_prefix][
                    constants.magnification_column_name_sheba
                ] = constants.magnification_column_name
                column_names[dataset_id_prefix][
                    constants.fold_column_name_sheba
                ] = constants.fold_column_name
                column_names[dataset_id_prefix][
                    constants.er_status_column_name_sheba
                ] = constants.er_status_column_name
                column_names[dataset_id_prefix][
                    constants.pr_status_column_name_sheba
                ] = constants.pr_status_column_name
                column_names[dataset_id_prefix][
                    constants.her2_status_column_name_sheba
                ] = constants.her2_status_column_name
                column_names[dataset_id_prefix][
                    constants.grade_column_name_sheba
                ] = constants.grade_column_name
                column_names[dataset_id_prefix][
                    constants.tumor_type_column_name_sheba
                ] = constants.tumor_type_column_name
                column_names[dataset_id_prefix][
                    constants.onco_ki_67_column_name_sheba
                ] = constants.onco_ki_67_column_name
                column_names[dataset_id_prefix][
                    constants.onco_score_11_column_name_sheba
                ] = constants.onco_score_11_column_name
                column_names[dataset_id_prefix][
                    constants.onco_score_18_column_name_sheba
                ] = constants.onco_score_18_column_name
                column_names[dataset_id_prefix][
                    constants.onco_score_26_column_name_sheba
                ] = constants.onco_score_26_column_name
                column_names[dataset_id_prefix][
                    constants.onco_score_31_column_name_sheba
                ] = constants.onco_score_31_column_name
                column_names[dataset_id_prefix][
                    constants.onco_score_all_column_name_sheba
                ] = constants.onco_score_all_column_name

        return column_names

    def _get_total_tiles_column_name(self):
        return f"Total tiles - {self._tile_size} compatible @ X{self._metadata_at_magnification}"

    def _get_legitimate_tiles_column_name(self):
        return f"Legitimate tiles - {self._tile_size} compatible @ X{self._metadata_at_magnification}"

    def _get_slide_tile_usage_column_name(self, dataset_id_prefix):
        if dataset_id_prefix == "ABCTB":
            return f"Slide tile usage [%] (for {self._tile_size}^2 Pix/Tile)"
        else:
            return f"Slide tile usage [%] (for {self._tile_size}^2 Pix/Tile) @ X{self._metadata_at_magnification}"

    def _load_metadata(self):
        padding = 40
        self._logger.info(
            msg=utils.generate_title_text(text=f"Metadata Generator Configuration")
        )
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="datasets_base_dir_path",
                value=self._datasets_base_dir_path,
                indentation=1,
                padding=padding,
            )
        )
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="metadata_enhancement_dir_path",
                value=self._metadata_enhancement_dir_path,
                indentation=1,
                padding=padding,
            )
        )
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="log_file_path",
                value=self._log_file_path,
                indentation=1,
                padding=padding,
            )
        )
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="tile_size", value=self._tile_size, indentation=1, padding=padding
            )
        )
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="desired_magnification",
                value=self._metadata_at_magnification,
                indentation=1,
                padding=padding,
            )
        )
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="dataset_ids",
                value=self._dataset_ids,
                indentation=1,
                padding=padding,
            )
        )

        dataset_paths_str = (
            f := lambda d: {k: f(v) for k, v in d.items()}
            if type(d) == dict
            else str(d)
        )(self._dataset_paths)
        dataset_paths_str_dump = json.dumps(dataset_paths_str, indent=8)
        self._logger.info(
            msg=utils.generate_captioned_bullet_text(
                text="dataset_paths",
                value=dataset_paths_str_dump,
                indentation=1,
                padding=padding,
                newline=True,
            )
        )

        self._logger.info(msg="")
        self._logger.info(msg=utils.generate_title_text(text=f"Metadata Processing"))
        df = None
        for _, dataset_id in enumerate(self._dataset_ids):
            self._logger.info(
                msg=utils.generate_captioned_bullet_text(
                    text="Processing Metadata For",
                    value=dataset_id,
                    indentation=1,
                    padding=padding,
                )
            )
            slide_metadata_file = os.path.join(
                self._dataset_paths[dataset_id],
                MetadataGenerator._get_slides_data_file_name(dataset_id=dataset_id),
            )
            grid_metadata_file = os.path.join(
                self._dataset_paths[dataset_id],
                MetadataGenerator._get_grids_folder_name(
                    desired_magnification=self._metadata_at_magnification
                ),
                constants.grid_data_file_name,
            )

            slide_df = pandas.read_excel(io=slide_metadata_file)
            slide_df = slide_df.filter(regex='^(?!Unnamed).*')
            # slide_df[constants.patient_barcode_column_name] = slide_df[constants.patient_barcode_column_name].astype(int)

            grid_df = pandas.read_excel(io=grid_metadata_file)
            grid_df = grid_df.filter(regex='^(?!Unnamed).*')

            current_df = pandas.DataFrame(
                {
                    **slide_df.set_index(keys=constants.file_column_name).to_dict(),
                    **grid_df.set_index(keys=constants.file_column_name).to_dict(),
                }
            )
            current_df.reset_index(inplace=True)
            current_df.rename(
                columns={"index": constants.file_column_name}, inplace=True
            )

            print(f'ROWS COUNT - after read_excel: {current_df.shape[0]}')
            current_df = self._rename_metadata(
                df=current_df,
                dataset_id_prefix=MetadataGenerator._get_dataset_id_prefix(
                    dataset_id=dataset_id
                ),
            )

            print(f'ROWS COUNT - after _rename_metadata: {current_df.shape[0]}')
            current_df = self._prevalidate_metadata(df=current_df)

            print(f'ROWS COUNT - after _prevalidate_metadata: {current_df.shape[0]}')
            current_df = self._enhance_metadata(df=current_df, dataset_id=dataset_id)

            print(f'ROWS COUNT - after _enhance_metadata: {current_df.shape[0]}')
            current_df = MetadataGenerator._select_metadata(df=current_df)

            print(f'ROWS COUNT - after _select_metadata: {current_df.shape[0]}')
            current_df = MetadataGenerator._standardize_metadata(df=current_df)

            print(f'ROWS COUNT - after _standardize_metadata: {current_df.shape[0]}')
            current_df = self._postvalidate_metadata(df=current_df)

            print(f'ROWS COUNT - after _postvalidate_metadata: {current_df.shape[0]}')
            if df is None:
                df = current_df
            else:
                df = pandas.concat((df, current_df))

        return df

    def _enhance_metadata_tcga(self, df: pandas.DataFrame):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)
        df = MetadataGenerator._add_NA_ki_67_status(df=df)
        df = MetadataGenerator._add_NA_onco_data(df=df)

        brca_tcga_pan_can_atlas_2018_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "TCGA",
                    "brca_tcga_pan_can_atlas_2018_clinical_data.tsv",
                )
            ),
            sep="\t",
        )

        brca_tcga_pub_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "TCGA",
                    "brca_tcga_pub_clinical_data.tsv",
                )
            ),
            sep="\t",
        )

        brca_tcga_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "TCGA",
                    "brca_tcga_clinical_data.tsv",
                )
            ),
            sep="\t",
        )

        brca_tcga_pub2015_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "TCGA",
                    "brca_tcga_pub2015_clinical_data.tsv",
                )
            ),
            sep="\t",
        )

        cell_genomics_tcga_file1_df = pandas.read_excel(
            io=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "TCGA",
                    "1-s2.0-S2666979X21000835-mmc2.xlsx",
                )
            )
        )

        cell_genomics_tcga_file2_df = pandas.read_excel(
            io=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "TCGA",
                    "1-s2.0-S2666979X21000835-mmc3.xlsx",
                )
            )
        )

        annotations_tcga = MetadataGenerator._extract_annotations(
            df=cell_genomics_tcga_file2_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_tcga,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_tcga,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_tcga,
            calculate_grade=MetadataGenerator._calculate_grade_tcga,
        )

        enhanced_metadata = pandas.concat([annotations_tcga])
        # df = pandas.merge(
        #     left=df,
        #     right=enhanced_metadata,
        #     on=[
        #         constants.patient_barcode_column_name,
        #         constants.slide_barcode_prefix_column_name,
        #     ],
        # )

        df[constants.patient_barcode_column_name] = df[constants.patient_barcode_column_name].astype(str)
        enhanced_metadata[constants.patient_barcode_column_name] = enhanced_metadata[constants.patient_barcode_column_name].astype(str)
        merged_df = pandas.merge(df, enhanced_metadata, on=[constants.patient_barcode_column_name, constants.slide_barcode_prefix_column_name], how='outer', indicator=True)
        df = merged_df.loc[merged_df['_merge'] != 'right_only']
        return df

    def _enhance_metadata_carmel_1_8(self, df):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)
        df = MetadataGenerator._add_NA_onco_data(df=df)

        carmel_annotations_Batch11_26_10_21_df = pandas.read_excel(
            io=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "Carmel",
                    "Carmel_annotations_Batch11_26-10-21.xlsx",
                )
            )
        )

        carmel_annotations_26_10_2021_df = pandas.read_excel(
            io=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path,
                    "Carmel",
                    "Carmel_annotations_26-10-2021.xlsx",
                )
            )
        )

        annotations1_carmel = MetadataGenerator._extract_annotations(
            df=carmel_annotations_Batch11_26_10_21_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_carmel,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_carmel,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_carmel,
            calculate_grade=MetadataGenerator._calculate_grade_carmel,
        )

        annotations2_carmel = MetadataGenerator._extract_annotations(
            df=carmel_annotations_26_10_2021_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_carmel,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_carmel,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_carmel,
            calculate_grade=MetadataGenerator._calculate_grade_carmel,
        )

        enhanced_metadata = pandas.concat([annotations1_carmel, annotations2_carmel])
        # try:
        # print(df.patient_barcode)
        # print(enhanced_metadata.patient_barcode)

        # Merge the two dataframes using the 'outer' merge method
        df[constants.patient_barcode_column_name] = df[constants.patient_barcode_column_name].astype(int)
        enhanced_metadata[constants.patient_barcode_column_name] = enhanced_metadata[constants.patient_barcode_column_name].astype(int)
        merged_df = pandas.merge(df, enhanced_metadata, on=[constants.patient_barcode_column_name, constants.slide_barcode_prefix_column_name], how='outer', indicator=True)
        df = merged_df.loc[merged_df['_merge'] != 'right_only']

        # df = pandas.merge(
        #     left=df,
        #     right=enhanced_metadata,
        #     on=[
        #         constants.patient_barcode_column_name,
        #         constants.slide_barcode_prefix_column_name,
        #     ],
        # )
        # except Exception:
        #     h = 5
        return df

    def _enhance_metadata_carmel_9_11(self, df):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)
        df = MetadataGenerator._add_NA_grade(df=df)
        df = MetadataGenerator._add_NA_tumor_type(df=df)
        df = MetadataGenerator._add_NA_onco_data(df=df)
        return df

    def _enhance_metadata_abctb(self, df):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)
        df = MetadataGenerator._add_NA_ki_67_status(df=df)
        df = MetadataGenerator._add_NA_onco_data(df=df)

        abctb_path_data_df = pandas.read_excel(
            io=os.path.normpath(
                os.path.join(
                    self._metadata_enhancement_dir_path, "ABCTB", "ABCTB_Path_Data.xlsx"
                )
            )
        )

        annotations_abctb = MetadataGenerator._extract_annotations(
            df=abctb_path_data_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_abctb,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_abctb,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_abctb,
            calculate_grade=MetadataGenerator._calculate_grade_abctb,
        )

        enhanced_metadata = pandas.concat([annotations_abctb])

        df[constants.patient_barcode_column_name] = df[constants.patient_barcode_column_name].astype(str)
        enhanced_metadata[constants.patient_barcode_column_name] = enhanced_metadata[constants.patient_barcode_column_name].astype(str)
        merged_df = pandas.merge(df, enhanced_metadata, on=[constants.patient_barcode_column_name, constants.slide_barcode_prefix_column_name], how='outer', indicator=True)
        df = merged_df.loc[merged_df['_merge'] != 'right_only']

        # df = pandas.merge(
        #     left=df,
        #     right=enhanced_metadata,
        #     on=[
        #         constants.patient_barcode_column_name,
        #         constants.slide_barcode_prefix_column_name,
        #     ],
        # )
        return df

    def _enhance_metadata_sheba(self, df):
        df = MetadataGenerator._add_NA_ki_67_status(df=df)
        return df

    def _enhance_metadata_haemek(self, df):
        df = MetadataGenerator._add_NA_onco_data(df=df)
        df = MetadataGenerator._add_NA_grade(df=df)
        return df

    def _enhance_metadata(self, df, dataset_id):
        if dataset_id == "TCGA":
            df = self._enhance_metadata_tcga(df=df)
        elif dataset_id.startswith("CARMEL"):
            if constants.get_dataset_id_suffix(dataset_id=dataset_id) < 9:
                df = self._enhance_metadata_carmel_1_8(df=df)
            else:
                df = self._enhance_metadata_carmel_9_11(df=df)
        elif dataset_id == "ABCTB":
            df = self._enhance_metadata_abctb(df=df)
        elif dataset_id.startswith("SHEBA"):
            df = self._enhance_metadata_sheba(df=df)
        elif dataset_id.startswith("HAEMEK"):
            df = self._enhance_metadata_haemek(df=df)

        df = self._add_tiles_count(df=df)
        return df

    def _rename_metadata(self, df, dataset_id_prefix):
        column_names = self._build_column_names()[dataset_id_prefix]
        df = df.rename(columns=column_names)
        return df

    def _prevalidate_metadata(self, df):
        df = df.dropna(subset=[
            constants.patient_barcode_column_name,
            constants.file_column_name,
            constants.fold_column_name,
            constants.mpp_column_name,
            constants.total_tiles_column_name,
            constants.width_column_name,
            constants.height_column_name,
            constants.dataset_id_column_name])

        if constants.bad_segmentation_column_name in df.columns:
            indices_of_slides_with_bad_seg = set(
                df.index[df[constants.bad_segmentation_column_name] == 1]
            )
        else:
            indices_of_slides_with_bad_seg = set()

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(
            list(all_indices - indices_of_slides_with_bad_seg)
        )

        return df.iloc[valid_slide_indices]

    def _postvalidate_metadata(self, df):
        indices_of_slides_without_grid = set(
            df.index[df[constants.tiles_count_column_name] == 0]
        )

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(
            list(all_indices - indices_of_slides_without_grid)
        )
        return df.iloc[valid_slide_indices]

    @staticmethod
    def _build_path_suffixes() -> Dict:
        path_suffixes = {
            constants.dataset_id_tcga: f"Breast/{constants.dataset_id_tcga}",
            constants.dataset_id_abctb: f"Breast/{constants.dataset_id_abctb}_TIF",
        }

        for i in range(1, 12):
            if i in range(1, 9):
                batches = "1-8"
            else:
                batches = "9-11"
            path_suffixes[
                f"{constants.dataset_id_carmel}{i}"
            ] = f"Breast/{constants.dataset_id_carmel.capitalize()}/{batches}/Batch_{i}/{constants.dataset_id_carmel}{i}"

        for i in range(2, 7):
            path_suffixes[
                f"{constants.dataset_id_sheba}{i}"
            ] = f"Breast/{constants.dataset_id_sheba.capitalize()}/Batch_{i}/{constants.dataset_id_sheba}{i}"

        for i in range(1, 4):
            path_suffixes[
                f"{constants.dataset_id_haemek}{i}"
            ] = f"Breast/{constants.dataset_id_haemek.capitalize()}/Batch_{i}/{constants.dataset_id_haemek.capitalize}{i}"

        return path_suffixes

    @staticmethod
    def _get_slides_data_file_name(dataset_id: str) -> str:
        return f"slides_data_{dataset_id}.xlsx"

    @staticmethod
    def _get_grids_folder_name(desired_magnification: int) -> str:
        return f"Grids_{desired_magnification}"

    @staticmethod
    def _get_dataset_id_prefix(dataset_id: str) -> str:
        return "".join(i for i in dataset_id if not i.isdigit())

    ############
    ### TCGA ###
    ############
    @staticmethod
    def _calculate_grade_tcga(row: pandas.Series) -> Union[str, int, pandas.NAType]:
        try:
            column_names = [
                "Epithelial tubule formation",
                "Nuclear pleomorphism",
                "Mitosis",
            ]
            grade_score = 0
            for column_name in column_names:
                column_score = re.findall(r"\d+", str(row[column_name]))
                if len(column_score) == 0:
                    return pandas.NA
                grade_score = grade_score + int(column_score[0])

            if 3 <= grade_score <= 5:
                return 1
            elif 6 <= grade_score <= 7:
                return 2
            else:
                return 3
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_tumor_type_tcga(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            column_name = "2016 Histology Annotations"
            tumor_type = row[column_name]

            if tumor_type == "Invasive ductal carcinoma":
                return "IDC"
            elif tumor_type == "Invasive lobular carcinoma":
                return "ILC"
            else:
                return "OTHER"
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_slide_barcode_prefix_tcga(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            return row[constants.patient_barcode_column_name_enhancement_tcga]
        except Exception:
            return pandas.NA

    #############
    ### ABCTB ###
    #############
    @staticmethod
    def _calculate_grade_abctb(row: pandas.Series) -> Union[str, int, pandas.NAtype]:
        try:
            column_name = "Histopathological Grade"
            column_score = re.findall(r"\d+", str(row[column_name]))
            if len(column_score) == 0:
                return pandas.NA

            return int(column_score[0])
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_tumor_type_abctb(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            column_name = "Primary Histologic Diagnosis"
            tumor_type = row[column_name]

            if tumor_type == "IDC":
                return "IDC"
            elif tumor_type == "ILC":
                return "ILC"
            else:
                return "OTHER"
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_slide_barcode_prefix_abctb(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            return row[constants.file_column_name_enhancement_abctb]
        except Exception:
            return pandas.NA

    ##############
    ### CARMEL ###
    ##############
    @staticmethod
    def _calculate_grade_carmel(row: pandas.Series) -> Union[str, int, pandas.NAType]:
        try:
            column_name = "Grade"
            column_score = re.findall(r"\d+(?:\.\d+)?", str(row[column_name]))
            if len(column_score) == 0:
                return pandas.NA

            return int(float(column_score[0]))
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_tumor_type_carmel(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            column_name = "TumorType"
            tumor_type = row[column_name]

            if tumor_type == "IDC":
                return "IDC"
            elif tumor_type == "ILC":
                return "ILC"
            else:
                return "OTHER"
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_slide_barcode_prefix_carmel(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            slide_barcode = row[constants.slide_barcode_column_name_enhancement_carmel]
            block_id = row[constants.block_id_column_name_enhancement_carmel]
            if math.isnan(block_id):
                block_id = 1

            slide_barcode = f"{slide_barcode.replace('/', '_')}_{int(block_id)}"
            return slide_barcode
        except Exception:
            return pandas.NA

    #############
    ### SHEBA ###
    #############
    @staticmethod
    def _calculate_grade_sheba(row: pandas.Series) -> Union[str, int, pandas.NAType]:
        try:
            column_name = "Grade"
            column_score = re.findall(r"\d+(?:\.\d+)?", str(row[column_name]))
            if len(column_score) == 0:
                return pandas.NA

            return int(float(column_score[0]))
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_tumor_type_sheba(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            column_name = "Histology"
            tumor_type = row[column_name]

            if tumor_type.startswith("IDC"):
                return "IDC"
            elif tumor_type.startswith("ILC"):
                return "ILC"
            else:
                return "OTHER"
        except Exception:
            return pandas.NA

    @staticmethod
    def _calculate_slide_barcode_prefix_sheba(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            return row[constants.patient_barcode_column_name]
        except Exception:
            return pandas.NA

    ###########
    ### ALL ###
    ###########
    def _get_tiles_count(self, row: pandas.Series) -> int:
        dataset_id = row[constants.dataset_id_column_name]
        dataset_path = self._dataset_paths[dataset_id]
        image_file_name_stem = Path(row[constants.file_column_name]).stem
        if utils.check_segmentation_data_exists(dataset_path=dataset_path, desired_magnification=self._metadata_at_magnification, image_file_name_stem=image_file_name_stem, tile_size=self._tile_size) is False:
            return 0
        else:
            segmentation_data = utils.load_segmentation_data(dataset_path=dataset_path, desired_magnification=self._metadata_at_magnification, image_file_name_stem=image_file_name_stem, tile_size=self._tile_size)
            return segmentation_data.shape[0]

    @staticmethod
    def _calculate_slide_barcode_prefix(row: pandas.Series) -> Union[str, pandas.NAType]:
        try:
            dataset_id = row[constants.dataset_id_column_name]
            if dataset_id == "TCGA":
                return row[constants.patient_barcode_column_name]
            elif dataset_id == "ABCTB":
                return row[constants.file_column_name].replace("tif", "ndpi")
            elif dataset_id.startswith("CARMEL"):
                return row[constants.slide_barcode_column_name_carmel][:-2]
            elif dataset_id == "SHEBA":
                return row[constants.patient_barcode_column_name]
        except Exception:
            return pandas.NA

    def _add_tiles_count(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df[constants.tiles_count_column_name] = df.apply(
            lambda row: self._get_tiles_count(row=row), axis=1
        )
        return df

    @staticmethod
    def _add_NA_tumor_type(
        df: pandas.DataFrame
    ) -> pandas.DataFrame:
        df[constants.tumor_type_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        return df

    @staticmethod
    def _add_NA_grade(
        df: pandas.DataFrame
    ) -> pandas.DataFrame:
        df[constants.grade_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        return df

    @staticmethod
    def _add_NA_ki_67_status(
        df: pandas.DataFrame
    ) -> pandas.DataFrame:
        df[constants.ki_67_status_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        return df

    @staticmethod
    def _add_NA_onco_data(
        df: pandas.DataFrame
    ) -> pandas.DataFrame:
        df[constants.onco_ki_67_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        df[constants.onco_score_11_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        df[constants.onco_score_18_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        df[constants.onco_score_26_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        df[constants.onco_score_31_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        df[constants.onco_score_all_column_name] = df.apply(
            lambda row: pandas.NA, axis=1
        )

        return df

    @staticmethod
    def _extract_annotations(
        df: pandas.DataFrame,
        patient_barcode_column_name: str,
        calculate_slide_barcode_prefix: Callable,
        calculate_tumor_type: Callable,
        calculate_grade: Callable,
    ) -> pandas.DataFrame:
        df[constants.slide_barcode_prefix_column_name] = df.apply(
            lambda row: calculate_slide_barcode_prefix(row), axis=1
        )
        df[constants.tumor_type_column_name] = df.apply(
            lambda row: calculate_tumor_type(row), axis=1
        )
        df[constants.grade_column_name] = df.apply(
            lambda row: calculate_grade(row), axis=1
        )

        annotations = df[
            [
                patient_barcode_column_name,
                constants.slide_barcode_prefix_column_name,
                constants.grade_column_name,
                constants.tumor_type_column_name,
            ]
        ]
        annotations = annotations.rename(
            columns={patient_barcode_column_name: constants.patient_barcode_column_name}
        )

        return annotations

    @staticmethod
    def _add_slide_barcode_prefix(df: pandas.DataFrame) -> pandas.DataFrame:
        df[constants.slide_barcode_prefix_column_name] = df.apply(
            lambda row: MetadataGenerator._calculate_slide_barcode_prefix(row), axis=1
        )
        return df

    @staticmethod
    def _select_metadata(df: pandas.DataFrame) -> pandas.DataFrame:
        df = df[
            [
                constants.file_column_name,
                constants.patient_barcode_column_name,
                constants.dataset_id_column_name,
                constants.mpp_column_name,
                constants.total_tiles_column_name,
                constants.tiles_count_column_name,
                constants.legitimate_tiles_column_name,
                constants.width_column_name,
                constants.height_column_name,
                constants.magnification_column_name,
                constants.er_status_column_name,
                constants.pr_status_column_name,
                constants.her2_status_column_name,
                constants.grade_column_name,
                constants.tumor_type_column_name,
                constants.ki_67_status_column_name,
                constants.onco_ki_67_column_name,
                constants.onco_score_11_column_name,
                constants.onco_score_18_column_name,
                constants.onco_score_26_column_name,
                constants.onco_score_31_column_name,
                constants.onco_score_all_column_name,
                constants.fold_column_name,
            ]
        ]
        return df

    @staticmethod
    def _standardize_metadata(df: pandas.DataFrame) -> pandas.DataFrame:
        # df = df.dropna()

        pandas.options.mode.chained_assignment = None
        # df = df[~df[constants.fold_column_name].isin(constants.invalid_values)]

        print(f'BEFORE invalid values removal: {df.shape[0]}')

        df = df[~df[[constants.fold_column_name, constants.patient_barcode_column_name, constants.file_column_name]].isin(constants.invalid_values).any(axis=1)]
        print(f'AFTER invalid values removal: {df.shape[0]}')

        folds = list(df[constants.fold_column_name].unique())
        numeric_folds = [utils.to_int(fold) for fold in folds]
        max_val = numpy.max(numeric_folds) + 1
        df.loc[
            df[constants.fold_column_name] == "test", constants.fold_column_name
        ] = max_val
        df[constants.fold_column_name] = df[constants.fold_column_name].astype(int)
        df = df.replace(constants.invalid_values, constants.invalid_value)
        # df = df.fillna(pandas.NA)
        # df = df.dropna()
        return df
