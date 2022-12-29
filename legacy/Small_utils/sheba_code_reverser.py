import pandas as pd
import numpy as np
import os
import re

dir_name = r'C:\ran_data\Sheba\5.4.22'
fn = r'04_04_2022_ran.xlsx'
out_fn = r'04_04_2022_out.xlsx'

df = pd.read_excel(os.path.join(dir_name, fn))

new_df = pd.DataFrame()

for row in df.iterrows():
    code_reversed = row[1]['Column1'].split('/')
    new_code = 'GS' + code_reversed[-1] + code_reversed[-2] + re.sub("[^0-9]", "", code_reversed[-3])[::-1]
    new_row = row[1]
    new_row['Code'] = new_code
    new_row['PatientID'] = row[0]
    new_df = new_df.append(new_row)

new_df.to_excel(os.path.join(dir_name, out_fn))
print('finished')