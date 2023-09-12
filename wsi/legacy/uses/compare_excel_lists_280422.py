from Small_utils.compare_excel_lists import compare_excel_lists
dir_name = r'C:\ran_data\Sheba\missing slides 280422'
fn1 = r'barcode_list_SHEBA_batch6.xlsx'
fn2 = r'Blocks_collected (1)_out.xlsx'

compare_excel_lists(dir_name, fn1, fn2, header_name='compare')