import os
from Dataset_Maker import dataset_utils
import glob


def remove_slides_according_to_list(in_dir, dataset):
    # delete slides (move to a different folder) according to file
    # assume we already have a "deleted slides" folder with an slides_to_delete.xlsx file in it
    print('removing slides according to "slides_to_delete" list')
    backup_ext = '_before_slide_delete'
    deleted_dir = os.path.join(in_dir, 'deleted slides')
    assert (os.path.isdir(deleted_dir)), "folder 'deleted slides' does not exist in input directory"
    excel_file = os.path.join(deleted_dir, 'slides_to_delete_' + dataset + '.xlsx')
    slides_to_delete_DF = dataset_utils.open_excel_file(excel_file)

    slides_data_file, slides_data_DF = dataset_utils.load_backup_slides_data(in_dir, dataset, extension=backup_ext)
    assert (are_all_duplicate_slides_listed(slides_data_DF, slides_to_delete_DF)), \
        'check slides_to_delete list - some duplicate slides are missing'

    grids_dirs = glob.glob(os.path.join(in_dir, dataset, 'Grids*'))
    grid_data_DF_list, grid_data_file_list = get_grid_data_file_list(grids_dirs, backup_ext)

    for row in slides_to_delete_DF.iterrows():
        slide_file, slide_barcode = get_slide_to_remove(in_dir, dataset, row[1]['slide'])
        if slide_file == "":
            continue
        try:
            slides_data_DF, grid_data_DF_list = remove_slide_and_metadata(in_dir, dataset, slide_file,
                                                                          deleted_dir, slide_barcode, grids_dirs,
                                                                          slides_data_DF, grid_data_DF_list)
        except PermissionError:
            print("Operation not permitted")
            # For other errors
        except OSError as error:
            print(error)

    dataset_utils.save_df_to_excel(slides_data_DF, slides_data_file)

    for ii, grid_data_DF in enumerate(grid_data_DF_list):
        dataset_utils.save_df_to_excel(grid_data_DF, grid_data_file_list[ii])

    print('finished removing slides')


def rename_duplicate_slides(in_dir, dataset):
    # rename slides after removing the excess slides
    # assume we already have a "deleted slides" folder with an slides_to_delete.xlsx file in it
    print('renaming slides after duplicated were deleted')
    backup_ext = '_before_duplicate_slide_rename'
    slides_data_file, slides_data_DF = dataset_utils.load_backup_slides_data(in_dir, dataset, extension=backup_ext)
    grids_dirs = glob.glob(os.path.join(in_dir, dataset, 'Grids*'))
    grid_data_DF_list, grid_data_file_list = get_grid_data_file_list(grids_dirs, backup_ext)

    for row in slides_data_DF.iterrows():
        slide_file = row[1]['file']
        slide_barcode, slide_ext = os.path.splitext(slide_file)
        needs_rename = (slide_barcode[-2] == '-') & (slide_barcode[-1].isdigit())
        if needs_rename:
            new_slide_barcode = slide_barcode[:-2]

            assert (not os.path.isfile(os.path.join(in_dir, dataset, new_slide_barcode + '.' + slide_ext))), \
                "cannot rename slide, slide without extension exists for slide " + slide_barcode

            try:
                slides_data_DF, grid_data_DF_list = \
                    rename_slide_and_metadata_to_remove_suffix(in_dir, dataset, slide_file,
                                                               slide_barcode, grids_dirs,
                                                               slides_data_DF, grid_data_DF_list)
            except PermissionError:
                print("Operation not permitted")
                # For other errors
            except OSError as error:
                print(error)

    dataset_utils.save_df_to_excel(slides_data_DF, slides_data_file)

    for ii, grid_data_DF in enumerate(grid_data_DF_list):
        dataset_utils.save_df_to_excel(grid_data_DF, grid_data_file_list[ii])

    print('finished renaming slides')


def remove_slide_and_metadata(in_dir, dataset, slide_file, deleted_dir, slide_barcode,
                              grids_dirs, slides_data_DF, grid_data_DF_list):
    remove_slide(in_dir, dataset, slide_file, deleted_dir)

    remove_segdata_images(in_dir, dataset, slide_barcode)

    remove_grid_files(grids_dirs, slide_barcode)

    slides_data_DF = remove_row_from_metadata(slides_data_DF, slide_file, 'file')

    for ii, grid_data_DF in enumerate(grid_data_DF_list):
        grid_data_DF_list[ii] = remove_row_from_metadata(grid_data_DF, slide_file, 'file')

    return slides_data_DF, grid_data_DF_list


def rename_slide_and_metadata_to_remove_suffix(in_dir, dataset, slide_file, slide_barcode,
                                               grids_dirs, slides_data_DF, grid_data_DF_list):
    rename_slide_to_remove_suffix(in_dir, dataset, slide_file)

    rename_segdata_images_to_remove_suffix(in_dir, dataset, slide_barcode)

    rename_grid_files_to_remove_suffix(grids_dirs, slide_barcode)

    slides_data_DF = rename_row_from_metadata_to_remove_suffix(slides_data_DF, slide_file, 'file')

    for ii, grid_data_DF in enumerate(grid_data_DF_list):
        grid_data_DF_list[ii] = rename_row_from_metadata_to_remove_suffix(grid_data_DF, slide_file, 'file')

    return slides_data_DF, grid_data_DF_list


def remove_slide(in_dir, dataset, slide_file, deleted_dir):

    # move slide file
    slide_file_full_path = os.path.join(in_dir, dataset, slide_file)
    os.rename(slide_file_full_path, os.path.join(deleted_dir, slide_file))

    # move slide folder
    if dataset_utils.is_mrxs(slide_file):
        move_mrxs_folder(in_dir, dataset, slide_file, deleted_dir)


def rename_slide_to_remove_suffix(in_dir, dataset, slide_file):

    # move slide file
    full_file = os.path.join(in_dir, dataset, slide_file)
    rename_file_to_remove_suffix(full_file)

    # move slide folder
    if dataset_utils.is_mrxs(slide_file):
        slide_dir = get_mrxs_dir(slide_file)
        rename_file_to_remove_suffix(os.path.join(in_dir, dataset, slide_dir))


def move_mrxs_folder(in_dir, dataset, slide_file, dst_dir):
    slide_dir = get_mrxs_dir(slide_file)
    dir_path = os.path.join(in_dir, dataset, slide_dir)
    assert (os.path.isdir(dir_path)), "mrxs slide directory does not exist"
    os.rename(dir_path, os.path.join(dst_dir, slide_dir))


def get_mrxs_dir(slide_file):
    return slide_file[:-5]


def get_slide_to_remove(in_dir, dataset, slide_barcode):
    matching_slides = glob.glob(os.path.join(in_dir, dataset, slide_barcode + '.*'))
    if len(matching_slides) == 0:
        print('slide ' + slide_barcode + 'not found in dataset')
        return "", ""
    assert (len(matching_slides) < 2), "found more than one match for slide " + slide_barcode
    matching_slide = matching_slides[0]
    slide_file = os.path.basename(matching_slide)
    return slide_file, slide_barcode


def remove_segdata_images(in_dir, dataset, slide_barcode):
    segdata_ext_list = ['_GridImage.png', '_thumb.png', '_SegMap.png', '_SegImage.png']
    for segdata_ext in segdata_ext_list:
        files_to_rename = glob.glob(os.path.join(in_dir, dataset, 'SegData', '*', slide_barcode + segdata_ext))
        for file in files_to_rename:
            dataset_utils.remove_file(file)


def rename_segdata_images_to_remove_suffix(in_dir, dataset, slide_barcode):
    segdata_ext_list = ['_GridImage.png', '_thumb.png', '_SegMap.png', '_SegImage.png']
    for segdata_ext in segdata_ext_list:
        files_to_delete = glob.glob(os.path.join(in_dir, dataset, 'SegData', '*', slide_barcode + segdata_ext))
        string_after_suffix = segdata_ext.split('.')[-2]
        for file in files_to_delete:
            rename_file_to_remove_suffix(file, string_after_suffix=string_after_suffix)


def remove_grid_files(grids_dirs, slide_barcode):
    for grid_dir in grids_dirs:
        files_to_delete = glob.glob(os.path.join(grid_dir, slide_barcode + '--tlsz*'))
        for file in files_to_delete:
            dataset_utils.remove_file(file)


def rename_grid_files_to_remove_suffix(grids_dirs, slide_barcode):
    for grid_dir in grids_dirs:
        files_to_rename = glob.glob(os.path.join(grid_dir, slide_barcode + '--tlsz*'))
        for file in files_to_rename:
            string_after_suffix = '--' + file.split('-')[-1].split('.')[-2]
            rename_file_to_remove_suffix(file, string_after_suffix=string_after_suffix)


def remove_row_from_metadata(slides_data_DF, value_to_remove, value_column='file'):
    return slides_data_DF[slides_data_DF[value_column] != value_to_remove]


def rename_row_from_metadata_to_remove_suffix(slides_data_DF, value_to_rename, value_column='file'):
    filename_no_suffix = get_filename_without_suffix(value_to_rename)
    slides_data_DF.loc[slides_data_DF[value_column] == value_to_rename, value_column] = filename_no_suffix
    return slides_data_DF


def get_grid_data_file_list(grids_dirs, backup_ext):
    grid_data_DF_list, grid_data_file_list = [], []
    for ii, grid_dir in enumerate(grids_dirs):
        grid_data_file_list.append(os.path.join(grid_dir, 'Grid_data.xlsx'))
        dataset_utils.backup_dataset_metadata(grid_data_file_list[ii], extension=backup_ext)
        grid_data_DF_list.append(dataset_utils.open_excel_file(grid_data_file_list[ii]))
    return grid_data_DF_list, grid_data_file_list


def rename_file_to_remove_suffix(file, string_after_suffix=''):
    fn_no_suffix = get_filename_without_suffix(file, string_after_suffix)
    os.rename(file, fn_no_suffix)


def get_filename_without_suffix(filename, string_after_suffix=''):
    fn, ext = os.path.splitext(filename)
    if string_after_suffix == '':
        fn_no_suffix = fn[:-2] + ext
    else:
        fn_no_suffix = fn[:-2-len(string_after_suffix)] + fn[-len(string_after_suffix):] + ext
    return fn_no_suffix


def are_all_duplicate_slides_listed(slides_data_DF, slides_to_delete_DF):
    # make sure that all the duplicated slides are listed
    # we will delete and rename slides and missing duplicates mean making a mess
    for row in slides_data_DF.iterrows():
        slide_file = row[1]['file']
        slide_barcode, slide_ext = os.path.splitext(slide_file)
        needs_rename = (slide_barcode[-2] == '-') & (slide_barcode[-1].isdigit())
        if needs_rename:
            slide_barcode_no_suffix = slide_barcode[:-2]

            # check if the slide exists on the delete list
            slide_file_exists = len(slides_to_delete_DF.loc[slides_to_delete_DF['slide'] == slide_barcode]) == 1
            slide_file_no_suffix_exists = len(slides_to_delete_DF.loc[slides_to_delete_DF['slide'] == slide_barcode_no_suffix]) == 1
            if not slide_file_exists and not slide_file_no_suffix_exists:
                return False
    return True
