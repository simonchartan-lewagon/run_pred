import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data
from run_pred.functions.c_feature_engineering import engineer_features


def balance_data(X_train_feat: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balances the input training dataset using random oversampling technique. The function first combines the
    features and target variables into a single DataFrame, and then performs oversampling on the minority class,
    which in this case is the gender feature. The balanced dataset is returned as two separate DataFrames,
    one containing the balanced feature set and the other containing the balanced target variable.

    Parameters:
    -----------
    X_train_feat: pd.DataFrame
        DataFrame containing the feature variables of the training dataset.
    y_train: pd.Series
        Series containing the target variable (time) of the training dataset.

    Returns:
    --------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing two DataFrames. The first DataFrame contains the balanced feature set, while the second
        DataFrame contains the corresponding target variable.
    """

    # Creating the training table from the input parameters
    train = X_train_feat.copy()
    train['time'] = y_train.copy()

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
    X_train_raw, X_test_raw, y_train_raw, y_test = split_data(dataset)
    X_train_feat = engineer_features(X_train_raw, y_train_raw)
    X_test_feat = engineer_features(X_test_raw, y_test)
    X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train_raw)

    print(X_train_balanced.shape)
    print(X_test_feat.shape)
    print(y_train_balanced.shape)
    print(y_test.shape)
