# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def dp2(file_name_train,file_name_test):
    '''
    This function loads and preprocess a training and test csv files
    :param file_name_train: Training csv file
    :param file_name_test: Testing csv file
    :return: X_train, X_test, y_train, y_test, column_names
    '''
    try:
        df_train = pd.read_csv(file_name_train, encoding="ISO-8859-1")
        df_test = pd.read_csv(file_name_test, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"Error: File '{file_name_train}' not found.")
        print(f"Error: File '{file_name_test}' not found.")
        return None

    X_train = df_train.iloc[:, :-1].to_numpy()
    X_test = df_test.iloc[:, :-1].to_numpy()

    #Standerdizing the data using Standardscaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    column_names = df_train.columns[:-1].tolist()


    y_train = LabelEncoder().fit_transform(df_train.iloc[:, -1])
    y_test = LabelEncoder().fit_transform(df_test.iloc[:, -1])

    return X_train_scaled, X_test_scaled, y_train, y_test, column_names #With normalization