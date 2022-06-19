from compare_excel_lists import fix_TMA_slide_names_in_label_excel_list
dir_name = r'C:\ran_data\TMA'
fn = r'slides_data_TMA_HE_01-011_labels.xlsx'

fix_TMA_slide_names_in_label_excel_list(dir_name, fn, filename_key='file')