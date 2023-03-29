# Install imbalanced-learn library if needed
# !pip install -U imbalanced-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data
from run_pred.functions.c_feature_engineering import engineer_features


def balance_data(X_train_feat, y_train):
    """
    This function uses the oversampling technique to balance each class of the gender variable.
    It takes as an input the training features dataset and the training target dataset and returns
    the corresponding balanced (oversampled) dataset for training purposes.

    Args:
        X_train (pandas.DataFrame) contains the features.
        y_train (pandas.Series) contains the target variable.
    """

    # Creating the training table from the input parameters
    train = X_train_feat
    train['time'] = y_train

    # Defining oversampling features and target variable
    X_bal = train.drop('gender', axis=1)
    y_bal = train['gender']

    # Executing the oversampling
    ros = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = ros.fit_resample(X_bal, y_bal)

    # Creating X_resampled et y_resampled
    dfb_train = X_resampled
    dfb_train['gender']= y_resampled

    # Redefining the training features dataset and target dataset, both balanced
    X_train_balanced = dfb_train.drop('time', axis=1).reset_index(drop=True)
    y_train_balanced = dfb_train['time'].reset_index(drop=True)

    return X_train_balanced, y_train_balanced


if __name__ == '__main__' :
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    X_train_raw, X_test, y_train, y_test = split_data(dataset)
    X_train_feat = engineer_features(X_train_raw)
    X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train)
    print(X_train_balanced.shape)
    print(y_train_balanced.shape)
