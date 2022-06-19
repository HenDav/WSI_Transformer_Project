from file_rename import rename_all_files_in_folder

dirname = r'C:\Users\User\OneDrive - Technion\Her2_batch1_w_control_review\marked'
delete_start = 'Inked'
delete_end = '_LI'
rename_all_files_in_folder(dirname, delete_start=delete_start, delete_end=delete_end)