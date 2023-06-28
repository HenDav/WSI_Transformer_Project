from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas
from tqdm import tqdm

from wsi_core import constants
from wsi_core.metadata import MetadataBase
from wsi_core.wsi import Slide, SlideContext, Tile, TilesManager


class SlidesManager(MetadataBase):
    def __init__(
        self,
        datasets_base_dir_path: Path,
        tile_size: int,
        desired_mpp: float,
        metadata_at_magnification: int,
        metadata_file_path: Path,
        row_predicate: Callable,  # [[pandas.DataFrame, ...], pandas.Index] somehow causes a bug, I have no idea why
        **predicate_args,
    ):
        if "folds" in predicate_args.keys():
            assert self.datasets_have_compatible_folds(
                predicate_args["datasets"]
            ), "datasets do not have compatible folds"
        self._desired_mpp = desired_mpp
        self._metadata_file_path = metadata_file_path
        self._slides = []
        self._current_slides = []
        # self._tile_to_slide_dict = self._create_tile_to_slide_dict()
        MetadataBase.__init__(
            self,
            datasets_base_dir_path=datasets_base_dir_path,
            tile_size=tile_size,
            metadata_at_magnification=metadata_at_magnification,
            desired_mpp=desired_mpp,
        )
        self._df = self._df.iloc[row_predicate(self._df, **predicate_args)]
        self._current_slides = self._create_slides()
        self._tiles_df = self._create_tiles_df()

        # self.start()
        # self.join()

    def __len__(self) -> int:
        return len(self._df)

    def _create_tiles_df(self) -> pandas.DataFrame:
        tiles_dfs = [slide._tiles_df.assign(slide_idx=idx).drop(columns=[TilesManager.tile_index,]) for idx, slide in enumerate(self._current_slides)]
        tiles_df = pandas.concat(tiles_dfs)
        return tiles_df
    
    def _create_slides(self) -> List[Slide]:
        slides = []
        # total = self._df.shape[0]
        row_index = 0
        for idx in tqdm(self._df.index, desc="Loading slides"):
            slide_context = SlideContext(
                row_index=row_index,
                metadata=self._df,
                dataset_paths=self._dataset_paths,
                desired_mpp=self._desired_mpp,
                tile_size=self._tile_size,
            )
            slide = Slide(slide_context=slide_context)
            # TODO: This is a temporary fix for the issue where the slide doesn't have interior tiles
            # if not slide.has_interior_tiles:
            #     self._df.drop(index=idx, inplace=True)
            #     continue
            slides.append(slide)
            row_index += 1
        # print(
        #     f"Loaded {len(slides)} slides, skipped {total - len(slides)} that have no interior tiles"
        # )

        # self._df = self._update_metadata()
        self._file_name_to_slide = self._create_file_name_to_slide_dict()
        # self.filter_folds(folds=None)

        return slides

    @staticmethod
    def datasets_have_compatible_folds(datasets: List[str]) -> bool:
        assert len(datasets) > 0, "datasets list should not be empty"
        compatible_folds = True
        datasets_folds = [constants.folds_for_datasets[dataset] for dataset in datasets]
        for dataset_folds in datasets_folds:
            compatible_folds = compatible_folds and datasets_folds[0] == dataset_folds
        return compatible_folds

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    @property
    def slides_count(self) -> int:
        return self._df.shape[0]
    
    @property
    def tiles_count(self) -> int:
        n_tiles = 0
        for slide in self._current_slides:
            n_tiles += slide.tiles_count
        return n_tiles

    # def get_slide_by_tile(self, tile: Tile) -> Slide:
    #     return self._tile_to_slide_dict[tile]

    def get_slides_ratio(self, ratio: float) -> List[Slide]:
        modulo = int(1 / ratio)
        return self.get_slides_modulo(modulo=modulo)

    def get_slides_modulo(self, modulo: int) -> List[Slide]:
        return self._slides[::modulo]

    def filter_folds(self, folds: Optional[List[int]]):
        if folds is not None:
            self._df = self._df[self._df[constants.fold_column_name].isin(folds)]
        else:
            self._df = self._df

        self._current_slides = self._get_slides()
        # self._slides_with_interior = self._get_slides_with_interior_tiles()

    def get_slide(self, slide_idx: int) -> Slide:
        return self._current_slides[slide_idx]
    
    def get_tile(self, tile_idx: int) -> Tuple[Slide, int]:
        row = self._tiles_df.iloc[[tile_idx]]
        slide_idx = row["slide_idx"].item()
        slide = self._current_slides[slide_idx]
        top_left_pixel = row[[TilesManager.pixel_x, TilesManager.pixel_y]].to_numpy()
        return Tile(slide.slide_context, top_left_pixel)

    def get_random_slide(self) -> Slide:
        index = np.random.randint(low=0, high=self._df.shape[0])
        return self.get_slide(index=index)

    def _add_shared_objects(self, namespace):
        namespace.metadata = self._df

    def _load_metadata(self) -> pandas.DataFrame:
        return pandas.read_csv(filepath_or_buffer=self._metadata_file_path)

    # def _generate_exempt_from_pickle(self) -> List[str]:
    #     # return []
    #     exempt_from_pickle = super()._generate_exempt_from_pickle()
    #     # exempt_from_pickle.append('_df')
    #     # exempt_from_pickle.append('_current_df')
    #     return exempt_from_pickle

    # def _post_join(self):
    #     self._slides = self._collect_slides()
    #
    #     # total_slides = 0
    #     # for slide in self._slides:
    #     #     total_slides = total_slides + slide.tiles_count
    #
    #     self._df = self._update_metadata()
    #     # # print(f'TASKS LEN {len(self._tasks)}')
    #     # # print(f'SLIDES LEN {len(self._slides)}')
    #     self._file_name_to_slide = self._create_file_name_to_slide_dict()
    #     self.filter_folds(folds=None)

    def _update_metadata(self) -> pandas.DataFrame:
        image_file_names = [
            slide.slide_context.image_file_name for slide in self._slides
        ]
        return self._df[self._df[constants.file_column_name].isin(image_file_names)]

    # def _collect_slides(self) -> List[Slide]:
    #     completed_tasks = cast(List[SlidesManagerTask], self._completed_tasks)
    #     return [task.slide for task in completed_tasks if task.slide is not None]

    # slides = []
    # for i, task in enumerate(self._completed_tasks):
    #     task = cast(SlidesManagerTask, task)
    #     if task.slide is not None:
    #         slides.append(task.slide)
    #
    # return slides

    # def _generate_tasks(self) -> List[ParallelProcessorTask]:
    #     tasks = []
    #
    #     combinations = list(itertools.product(*[
    #         [*range(self._df.shape[0])],
    #         [self._dataset_paths],
    #         [self._metadata_at_magnification],
    #         [self._tile_size]]))
    #
    #     for combination in combinations:
    #         tasks.append(SlidesManagerTask(
    #             row_index=combination[0],
    #             dataset_paths=combination[1],
    #             desired_magnification=combination[2],
    #             tile_size=combination[3]))
    #
    #     return tasks

    def _create_file_name_to_slide_dict(self) -> Dict[str, Slide]:
        file_name_to_slide = {}
        # print(f'slides len {len(self._slides)}')
        for slide in self._slides:
            # print(slide.slide_context.image_file_name)
            file_name_to_slide[slide.slide_context.image_file_name] = slide

        return file_name_to_slide

    def _create_tile_to_slide_dict(self) -> Dict[Tile, Slide]:
        tile_to_slide = {}
        for slide in self._slides:
            for tile in slide.tiles:
                tile_to_slide[tile] = slide

        return tile_to_slide

    def _get_slides(self) -> List[Slide]:
        # print(self._df[constants.file_column_name])
        # print(self._file_name_to_slide)
        return [
            self._file_name_to_slide[x] for x in self._df[constants.file_column_name]
        ]