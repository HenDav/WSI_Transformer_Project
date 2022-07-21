import pandas as pd

scanned_slides_file = r'slides_extracted.xlsx'
image_barcode_file = r'C:\ran_data\RAMBAM\SlideID_images2\out\barcode_list.csv'
scanned_slides = pd.read_excel(scanned_slides_file, sheet_name='list')
image_barcodes = pd.read_csv(image_barcode_file)
image_barcodes = image_barcodes.drop(['left','top', 'h', 'w'], axis=1)

merged_df = scanned_slides.merge(image_barcodes, how='outer', on='barcode')

merged_df.to_csv('slide_comparison.csv')
print('Finished')
