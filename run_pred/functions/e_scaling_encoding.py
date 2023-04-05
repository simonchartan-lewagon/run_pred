import pandas as pd
import numpy as np
import time
import os
import joblib

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data
from run_pred.functions.c_feature_engineering import engineer_features
from run_pred.functions.d_balancing import balance_data


def scale_encode_data(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: object = StandardScaler()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale and encode data using a given scaler.

    Args:
    X_train (pd.DataFrame): The training dataset
    X_test (pd.DataFrame): The testing dataset.
    scaler (object, optional): The scaler to use to scale the data. Defaults to StandardScaler().

    Returns:
    tuple: A tuple containing the scaled and encoded training and testing datasets.
    """

    X_test = pd.DataFrame(X_test,columns = X_train.columns)
    X_train_scaled, X_test_scaled = scale_features(X_train=X_train, X_test=X_test, scaler = scaler)
    X_train_scaled_encoded, X_test_scaled_encoded = encode_features(X_train=X_train_scaled, X_test=X_test_scaled)

    return X_train_scaled_encoded, X_test_scaled_encoded


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: object = StandardScaler(), save: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    if save:
        # Saves the fitted ohe parameters on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{scaler_name}_{timestamp}.joblib"
        print('scaler saving initiated')
        model_name = type(scaler).__name__

        # save ohe locally
        model_path = os.path.join("models", f"{model_name}.joblib")
        joblib.dump(scaler, open(model_path, 'wb'))
        print('fitted scaler parameters saved locally')

    return X_train_scaled, X_test_scaled

def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame, save: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Encode the gender feature using One-Hot-Encoding.

    Args:
    X_train (pd.DataFrame): training dataset.
    X_test (pd.DataFrame): testing dataset.
    save (bool, optional): Whether to save the fitted OneHotEncoder parameters on the hard drive. Defaults to False.

    Returns:
    tuple: A tuple of pd.DataFrame objects, with encoded categorical features. The first element is the encoded training dataset and the second is the encoded testing dataset.
    '''

    # Extract the column containing categorical features
    features_ohe = ['gender']

    # Extract the categorical data from the training and testing dataset
    X_train_cat = X_train[features_ohe]
    X_test_cat = X_test[features_ohe]

    # Initialize a OneHotEncoder object with parameters to handle binary variables and unknown values
    ohe = OneHotEncoder(
        drop = 'if_binary',
        sparse_output = False,
        handle_unknown = "ignore"
        )

    # Transform the categorical data of the training dataset using the OneHotEncoder object
    X_train.gender = pd.DataFrame(
        ohe.fit_transform(X_train_cat),
        columns = ohe.get_feature_names_out()
        )

    # Transform the categorical data of the testing dataset using the fitted OneHotEncoder object
    X_test.gender = pd.DataFrame(
        ohe.transform(X_test_cat),
        columns = ohe.get_feature_names_out()
        )

    if save:
        # Saves the fitted ohe parameters on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{scaler_name}_{timestamp}.joblib"
        print('ohe saving initiated')
        model_name = type(ohe).__name__

        # save ohe locally
        model_path = os.path.join("models", f"{model_name}.joblib")
        joblib.dump(ohe, open(model_path, 'wb'))
        print('fitted ohe parameters saved locally')

    return X_train, X_test



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
