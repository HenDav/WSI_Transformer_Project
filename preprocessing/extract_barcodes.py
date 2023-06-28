# standard library
from pathlib import Path
import argparse

# wsi
from barcode_extraction import BarcodeExtractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--data-dir-path', type=str)
    args = parser.parse_args()

    barcode_extractor = BarcodeExtractor(dataset_name=args.dataset_name, data_dir_path=Path(args.data_dir_path))
    barcode_extractor.extract_barcodes()
