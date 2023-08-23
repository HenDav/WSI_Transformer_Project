# python peripherals
from pathlib import Path
import argparse

# gipmed
from core.metadata import MetadataGenerator
from tap import Tap
from typing import List

# gipmed
from core.metadata import MetadataGenerator

class MetadataGeneratorArgumentsParser(Tap):
    datasets_base_dir_path: Path
    tile_size: int
    desired_magnification: int
    metadata_enhancement_dir_path: Path
    output_dir_path: Path
    dataset_ids: List[str]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--datasets-base-dir-path', type=str)
    parser.add_argument('--tile-size', type=int)
    parser.add_argument('--desired-magnification', type=int)
    parser.add_argument('--metadata-enhancement-dir-path', type=str)
    parser.add_argument('--output-dir-path', type=str)
    parser.add_argument('--dataset-ids', type=str, nargs='+')
    args = parser.parse_args()

    metadata_generator = MetadataGenerator(
        name='metadata_generation_app',
        datasets_base_dir_path=Path(args.datasets_base_dir_path),
        tile_size=args.tile_size,
        desired_magnification=args.desired_magnification,
        metadata_enhancement_dir_path=Path(args.metadata_enhancement_dir_path),
        output_dir_path=Path(args.output_dir_path),
        dataset_ids=args.dataset_ids
    )

    metadata_generator.save_metadata()
