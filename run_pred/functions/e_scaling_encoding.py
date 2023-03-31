# Importation des librairies nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data
from run_pred.functions.c_feature_engineering import engineer_features
from run_pred.functions.d_balancing import balance_data


def scale_encode_data(X_train, X_test, scaler = StandardScaler()):

    X_test = pd.DataFrame(X_test,columns = X_train.columns)
    X_train_scaled, X_test_scaled = scale_features(X_train=X_train, X_test=X_test, scaler = scaler)
    X_train_scaled_encoded, X_test_scaled_encoded = encode_features(X_train=X_train_scaled, X_test=X_test_scaled)

    return X_train_scaled_encoded, X_test_scaled_encoded


def scale_features(X_train, X_test, scaler = StandardScaler()):
    """
    This function applies the variable scaling method to the numerical variables of a dataset. It returns both datasets with the numerical variables scaled.

    Args:
    X_train (pandas.DataFrame): The training dataset.
    X_test (pandas.DataFrame): The test dataset.

    Returns:
    (pandas.DataFrame, pandas.DataFrame): A tuple of two DataFrames containing the scaled training and test datasets, respectively, with the numerical variables scaled
    """

	# Float vs other variables --> Float variables are the only ones to scale,
    # others are variables that are either pre-encoded or to encode

    X_train_incl_num = X_train[['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km']]
    X_test_incl_num = X_test[['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km']]

    X_train_excl_num = X_train.drop(columns = ['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km'])
    X_test_excl_num = X_test.drop(columns = ['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km'])

	# Applying the chosen scaling method
    scaler = scaler
    X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_incl_num),columns= X_train_incl_num.columns)
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_incl_num),columns= X_test_incl_num.columns)

    # Concaténation des variables numériques et non numériques
    X_train_scaled = X_train_excl_num.join(X_train_num_scaled, on = X_train_excl_num.index)
    X_test_scaled = X_test_excl_num.join(X_test_num_scaled, on = X_test_excl_num.index)

    return X_train_scaled, X_test_scaled

def encode_features(X_train, X_test):

    features_ohe = ['gender']

    X_train_cat = X_train[features_ohe]
    X_test_cat = X_test[features_ohe]

    ohe = OneHotEncoder(
        drop = 'if_binary',
        sparse_output = False,
        handle_unknown = "ignore"
        )

    X_train.gender = pd.DataFrame(
        ohe.fit_transform(X_train_cat),
        columns = ohe.get_feature_names_out()
        )

    X_test.gender = pd.DataFrame(
        ohe.transform(X_test_cat),
        columns = ohe.get_feature_names_out()
        )

    return X_train, X_test


def scale_target(y_train,y_test):
    """
    This function applies the StandardScaler method to center and scale the numerical data of the target variable y_train and y_test, and returns a new DataFrame with the transformed data.

    Args:
    y_train (pandas.DataFrame): The DataFrame containing the training data of the target variable.
    y_test (pandas.DataFrame): The DataFrame containing the test data of the target variable.

    Returns:
    pandas.DataFrame: A new DataFrame with the centered and scaled numerical data.
    """
    df_y_train = pd.DataFrame(y_train)
    df_y_test = pd.DataFrame(y_test)

    # Application de la méthode StandardScaler aux variables numériques
    scaler = StandardScaler()
    y_train_scaled = pd.DataFrame(scaler.fit_transform(df_y_train),columns=['time'])
    y_test_scaled = pd.DataFrame(scaler.transform(df_y_test),columns=['time'])

    return y_train_scaled, y_test_scaled


if __name__ == '__main__' :
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    X_train_raw, X_test_raw, y_train, y_test = split_data(dataset)
    X_train_feat = engineer_features(X_train_raw, y_train)
    X_test_feat = engineer_features(X_test_raw, y_test)
    X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train)
    X_train_balanced_scaled_encoded, X_test_scaled_encoded = scale_encode_data(X_train_balanced, X_test_feat)

    print(X_train_balanced_scaled_encoded.shape)
    print(X_test_scaled_encoded.shape)
    print(y_train_balanced.shape)
    print(y_test.shape)

    print(X_train_balanced_scaled_encoded.gender.isna().sum())
    print(X_test_scaled_encoded.gender.isna().sum())
    print(X_train_balanced_scaled_encoded.info())
    print(X_test_scaled_encoded.info())
