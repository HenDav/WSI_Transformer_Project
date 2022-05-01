import pandas as pd
import os


def save_metadata_file(df, dir_name, fn):
    out_fn = os.path.splitext(fn)[0] + '_out.xlsx'
    df.to_excel(os.path.join(dir_name, out_fn))


def add_new_row_to_metadata(df, code, metadata_row, column_name):
    code_strip = code.strip()
    new_row = metadata_row[1]
    new_row[column_name] = code_strip
    new_row['PatientID'] = metadata_row[0]
    df = df.append(new_row)
    return df


def parse_metadata_slides_to_separate_lines(dir_name, fn, column_name='CODE'):
    df = pd.read_excel(os.path.join(dir_name, fn))
    new_metadata_df = pd.DataFrame()
    for row in df.iterrows():
        code_list = row[1][column_name].split(';')
        for code in code_list:
            new_metadata_df = add_new_row_to_metadata(new_metadata_df, code, row, column_name)

    save_metadata_file(new_metadata_df, dir_name, fn)


def reversed_sheba_code(code):
    code_no_zeros = code.replace("-0", "-")
    reversed_code = code_no_zeros[::-1]
    reversed_code_numeral = reversed_code.replace("/", "").replace("-", "")
    return reversed_code_numeral


def decode_sheba_codes(dir_name, fn, column_name='CODE'):
    df = pd.read_excel(os.path.join(dir_name, fn))
    new_column_name = column_name + "_reversed"
    reversed_code_list = []
    for row in df.iterrows():
        code = row[1][column_name]
        reversed_code_list.append(reversed_sheba_code(code))
    df[new_column_name] = reversed_code_list
    save_metadata_file(df, dir_name, fn)