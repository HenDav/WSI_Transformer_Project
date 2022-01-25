import pandas as pd
import numpy as np
import os

dir_name = r'C:\ran_data\Sheba'
#fn = r'SHEBA ONCOTYPE 30_12_2021.xlsx'
fn = r'SHEBA ONCOTYPE 160122_Ran.xlsx'
out_fn = r'SHEBA ONCOTYPE 30_12_2021_Ran_out.xlsx'
barcode_list_fn = r'barcode_list_full.xlsx'

df = pd.read_excel(os.path.join(dir_name, fn))

barcode_list_df = pd.read_excel(os.path.join(dir_name, barcode_list_fn))
barcode_array = np.array(barcode_list_df['file'])
new_df = pd.DataFrame()
missing_in_metadata = []
missing_in_barcode_list = []
for row in df.iterrows():
    code_list = row[1]['Code'].split(';')
    for code in code_list:
        code_strip = code.strip()
        if not code_strip + '.tiff' in barcode_array:
            missing_in_barcode_list.append(code_strip)
        new_row = row[1]
        new_row['Code'] = code_strip
        #new_row['file'] = code.strip() + '.tiff'
        new_row['PatientID'] = row[0]
        new_df = new_df.append(new_row)

print('missing in barcode list:')
print(missing_in_barcode_list)

#look for slides that do not have metadata
code_array = np.array(new_df['Code'])
for barcode in barcode_array:
    if not os.path.splitext(barcode)[0] in code_array:
        missing_in_metadata.append(barcode)

print('missing in metadata file:')
print(missing_in_metadata)

missing_in_metadata_df = pd.DataFrame(missing_in_metadata)
missing_in_metadata_df.to_excel(os.path.join(dir_name, 'missing_in_metadata.xlsx'))

missing_in_barcode_list_df = pd.DataFrame(missing_in_barcode_list)
missing_in_barcode_list_df.to_excel(os.path.join(dir_name, 'missing_in_barcode_list.xlsx'))

#new_df.to_excel(os.path.join(dir_name, out_fn))
print('finished')