from Dataset_Maker import slide_walker, slide_rename, barcode2excel
import os

def prepare_dataset_for_training(Dataset, data_dir, scan_barcodes, get_slide_labels, step):
    if step == 0:
        prepare_dataset_step0(data_dir, Dataset, scan_barcodes, get_slide_labels)

    elif step == 1:
        prepare_dataset_step1(data_dir, Dataset)

    elif step == 2:
        prepare_dataset_step2()

    elif step == 3:
        prepare_dataset_step3()


def prepare_dataset_step0(data_dir, Dataset, scan_barcodes, get_slide_labels):
    slide_walker.create_slide_list(data_dir, Dataset)

    if get_slide_labels:
        slide_walker.add_barcodes_to_slide_list(data_dir, Dataset, scan_barcodes)

        if get_num_unreadable_labels_found(data_dir):
            barcode2excel.add_unreadable_label_images_to_slide_list(data_dir)


def prepare_dataset_step1(data_dir, Dataset):
    slide_rename.add_slide_rename_to_barcode_list(data_dir, Dataset)

    slide_rename.rename_slides_according_to_list(data_dir, Dataset)

    slide_rename.delete_empty_folders()

def prepare_dataset_step2():
    pass #TODO

def prepare_dataset_step3():
    pass #TODO


def get_num_unreadable_labels_found(data_dir):
    num = os.listdir(os.path.join(data_dir, 'unreadable_labels'))
    return num


'''def dataset_has_non_scannable_barcodes(Dataset):
    return Dataset[:4] == 'Her2' '''


def dataset_is_haemek(Dataset):
    return 'HAEMEK' in Dataset