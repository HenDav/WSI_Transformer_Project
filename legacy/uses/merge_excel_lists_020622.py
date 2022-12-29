from Small_utils.compare_excel_lists import merge_excel_lists
dir_name = r'C:\ran_data\BoneMarrow\AML'
fn1 = r'slides_data_AML.xlsx'
fn2 = r'AML_metadata.xlsx'

merge_excel_lists(dir_name, fn1, fn2, merge_key='file')