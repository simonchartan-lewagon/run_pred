# Importation des librairies nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Standard
def scaler_features(X_train, X_test):
    """
    This function applies the variable scaling method to the numerical variables of a dataset. It returns both datasets with the numerical variables scaled.

    Args:
    X_train (pandas.DataFrame): The training dataset.
    X_test (pandas.DataFrame): The test dataset.

    Returns:
    (pandas.DataFrame, pandas.DataFrame): A tuple of two DataFrames containing the scaled training and test datasets, respectively, with the numerical variables scaled

    """
	# Sélection des variables numériques
    X_train_float = X_train.select_dtypes(include='float')
    X_test_float = X_test.select_dtypes(include='float')

	# Sélection des variables non numériques et réinitialisation de l'index
    X_train_ex = X_train.select_dtypes(exclude='float').reset_index()
    X_test_ex = X_test.select_dtypes(exclude='float').reset_index()

	# Application de la méthode StandardScaler aux variables numériques
    scaler = StandardScaler()
    X_train_float_scaller = pd.DataFrame(scaler.fit_transform(X_train_float),columns= X_train_float.columns)
    X_test_float_scaller = pd.DataFrame(scaler.transform(X_test_float), columns= X_test_float.columns)

	# Concaténation des variables numériques et non numériques
    X_train_scaller = pd.concat([X_train_ex, X_train_float_scaller],axis=1)
    X_test_scaller =  pd.concat([X_test_ex, X_test_float_scaller])
    X_train_scaller = X_train_scaller.drop(columns='index')
    X_test_scaller = X_test_scaller.drop(columns='index')

    return X_train_scaller, X_test_scaller




def scaler_target(y_train, y_test):
    """
    This function applies the StandardScaler method to center and scale the numerical data of the target variable y_train and y_test, and returns a new DataFrame with the transformed data.

    Args:
    y_train (pandas.DataFrame): The DataFrame containing the training data of the target variable.
    y_test (pandas.DataFrame): The DataFrame containing the test data of the target variable.

    Returns:
    pandas.DataFrame: A new DataFrame with the centered and scaled numerical data.
    """
    df_y_train = pd.DataFrame(y_train).reset_index().drop(columns='index')
    df_y_test = pd.DataFrame(y_test).reset_index().drop(columns='index')

    # Application de la méthode StandardScaler aux variables numériques
    scaler = StandardScaler()
    y_train_scaller = pd.DataFrame(scaler.fit_transform(df_y_train),columns=['time'])
    y_test_scaller = pd.DataFrame(scaler.transform(df_y_test),columns=['time'])

    return y_train_scaller, y_test_scaller
