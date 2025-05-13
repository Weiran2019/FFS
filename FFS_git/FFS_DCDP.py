# -*- coding: utf-8 -*-
"""
Generating Table I

Referfence A. A. Mamun, H. Al-Sahaf, I. Welch, M. Barcellos and S. Camtepe, 
``Limitations of Advanced Persistent Threat Datasets: Insights for Cybersecurity Research,'' 
in 2024 34th International Telecommunication Networks and Applications Conference (ITNAC), 
Sydney, Australia, 2024.
"""

import csv
import pandas as pd
import numpy as np
from collections import Counter

def create_ab_csv(all_valid_rows,output_file_path):
    with open(output_file_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(all_valid_rows)

def dc1(df):
    all_valid_rows = []
    lines_removed_count = 0
    num_columns = len(df.columns)
    all_valid_rows.append(df.columns.tolist())
    for index, row in df.iterrows():
        if len(row) == num_columns:
            all_valid_rows.append(row)
        else:
            lines_removed_count += 1
    return all_valid_rows, lines_removed_count

def dc23dp1(df):
    '''
    This function used for removing unnecessary features, duplicates and inf values.
    :param df: Dataset
    :return: df
    '''
    # Replace spaces and slashes (\) with underscores (_) in column names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    # list of columns to drop.
    drop_cols = ['Flow_ID', 'Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port','Protocol','Timestamp']
    # list of columns to keep
    # keep_cols = list(set(range(len(df.columns))) - set(drop_cols))
    keep_cols = list(set(df.columns) - set(drop_cols))
    # Select only the columns to keep
    df = df[keep_cols]
    # Identify and count duplicate rows
    duplicate_count = df.duplicated().sum()
    # Count the number of duplicate rows
    print(f"Number of duplicate rows: {duplicate_count}")
    # Identify duplicate rows in the dataframe
    mask = df.duplicated()
    # Keep only the non-duplicated rows by using the inverted mask
    df = df.loc[~mask]
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Count the number of rows with NaN values
    num_nan_rows = df.isna().any(axis=1).sum()
    print(f"Number of rows with NaN values: {num_nan_rows}")
    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Move 'Label' column to the end
    label_col = df.pop('Label')
    df['Label'] = label_col

    # Replace negative values with zero in all columns except the 'Label' column
    df.iloc[:, :-1] = df.iloc[:, :-1].where(df.iloc[:, :-1] >= 0, 0)  # Efficient vectorized approach
    # Identify columns with only zero values
    zero_value_cols = df.columns[(df == 0).all()]
    # Print the number of columns with only zero values and their names
    print(f"Number of columns with only zero values: {len(zero_value_cols)}")
    if len(zero_value_cols) > 0:
        print("Columns with only zero values:", zero_value_cols.tolist())
    # Drop columns with only zeros.
    # df = df.loc[:, (df != 0).any(axis=0)]
    return df,zero_value_cols.tolist()


df_train = pd.read_csv("./data/Training.csv")
df_test = pd.read_csv("./data/Testing.csv")

all_valid_rows, lines_removed_count = dc1(df_train)
create_ab_csv(all_valid_rows, "./data/train_dc1.csv")
print(f"Number of lines removed: {lines_removed_count}")

all_valid_rows, lines_removed_count = dc1(df_test)
create_ab_csv(all_valid_rows, "./data/test_dc1.csv")
print(f"Number of lines removed: {lines_removed_count}")

train = pd.read_csv("./data/train_dc1.csv")
test = pd.read_csv("./data/test_dc1.csv")

train,tr_zero_cols = dc23dp1(train)
train = train.drop(tr_zero_cols,axis=1)
test, te_zero_cols = dc23dp1(test)
test = test.drop(tr_zero_cols,axis=1)

print(f"train: {train}")
print(f"test: {test}")
traincount = Counter(train.iloc[:,-1])
print(traincount)
testcount = Counter(test.iloc[:,-1])
print(testcount)
train.to_csv("./data/train_dc23dp1.csv",index=False)
test.to_csv("./data/test_dc23dp1.csv",index=False)


