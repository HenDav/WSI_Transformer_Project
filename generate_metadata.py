# python peripherals
from pathlib import Path

# gipmed
from wsi_core.metadata import MetadataGenerator, MetadataGeneratorArgumentsParser

if __name__ == '__main__':
    metadata_generator_arguments = MetadataGeneratorArgumentsParser().parse_args()

    metadata_generator = MetadataGenerator(
        name='metadata_generation_app',
        datasets_base_dir_path=metadata_generator_arguments.datasets_base_dir_path,
        tile_size=metadata_generator_arguments.tile_size,
        desired_magnification=metadata_generator_arguments.desired_magnification,
        metadata_enhancement_dir_path=metadata_generator_arguments.metadata_enhancement_dir_path,
        output_dir_path=metadata_generator_arguments.output_dir_path,
        dataset_ids=metadata_generator_arguments.dataset_ids
    )

    metadata_generator.save_metadata()
