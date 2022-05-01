from compare_excel_lists import compare_excel_lists
dir_name = r'C:\ran_data\Sheba\missing slides in conversion'
fn1 = r'barcode_list_batch6.xlsx'
fn2 = r'barcode_list_isyntax_new.xlsx'

compare_excel_lists(dir_name, fn1, fn2, header_name='file')