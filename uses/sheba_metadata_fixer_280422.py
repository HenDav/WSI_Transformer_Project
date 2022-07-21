from Small_utils.sheba_metadata_fixer import parse_metadata_slides_to_separate_lines
dir_name = r'C:\ran_data\Sheba'
fn = r'26_04_2022 data.xlsx'

parse_metadata_slides_to_separate_lines(dir_name, fn, column_name='CODE')