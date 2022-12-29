import os


def rename_all_files_in_folder(dirname, delete_start='', delete_end='', add_start='', add_end=''):
    file_list = os.listdir(dirname)
    start_len = len(delete_start)
    end_len = len(delete_end)
    for file in file_list:
        file_no_ext, ext = os.path.splitext(file)
        new_file_no_ext = file_no_ext
        if file_no_ext[:start_len] == delete_start:
            new_file_no_ext = new_file_no_ext[start_len:]
        if file_no_ext[-end_len:] == delete_end:
            new_file_no_ext = new_file_no_ext[:-end_len]

        new_file_no_ext = add_start + new_file_no_ext + add_end

        if new_file_no_ext != file_no_ext:
            os.rename(os.path.join(dirname, file), os.path.join(dirname, new_file_no_ext + ext))
