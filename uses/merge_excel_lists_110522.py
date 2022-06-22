from compare_excel_lists import merge_excel_lists
dir_name = r'C:\ran_data\TMA'
fn1 = r'slides_data_TMA_HE_01-011_no_labels.xlsx'
fn2 = r'slides_data_TMA_HE_01-011_labels.xlsx'

merge_excel_lists(dir_name, fn1, fn2, merge_key='file')