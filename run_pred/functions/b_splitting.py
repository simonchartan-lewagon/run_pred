import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from run_pred.functions.a_cleaning import clean_data

def split_data(run, test_size = 0.2):
    """
    This function creates raw train/test splits in the previously cleaned dataset.
    """
    X = run.drop(columns = ['time'])
    y = run.time

    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=1
        )

    # Resetting indexes to prevent conflicts further down the workflow
    X_train_raw = X_train_raw.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_train_raw = y_train_raw.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)

    return X_train_raw, X_test, y_train_raw, y_test

if __name__ == '__main__' :

    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    X_train_raw, X_test, y_train_raw, y_test = split_data(dataset)

    for data in [X_train_raw, X_test, y_train_raw, y_test]:
        assert(data.shape[0] == data.index.max()+1)

    print(X_train_raw.shape)
    print(X_test.shape)
    print(X_train_raw.shape)
    print(y_test.shape)
    print(X_train_raw.head())
