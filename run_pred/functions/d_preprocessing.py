# Importation des librairies nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Standard
def scaler_standard(X_train, X_test):
    """
    Cette fonction applique la méthode de mise à l'échelle des variables (scaling) sur les variables numériques d'un jeu de données.
    Elle retourne les deux jeux de données avec les variables numériques mises à l'échelle.

    Args:
        X_train (pandas.DataFrame): Le jeu de données d'entraînement.
        X_test (pandas.DataFrame): Le jeu de données de test.

    Returns:
        (pandas.DataFrame, pandas.DataFrame): Un tuple de deux DataFrames contenant les jeux de données d'entraînement et de test
        respectivement avec les variables numériques mises à l'échelle.

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




def scaler_standard(y_train, y_test):
    """
    Cette fonction applique la méthode StandardScaler pour centrer et réduire les données numériques de la variable cible
    y_train et y_test, et retourne un nouveau DataFrame avec les données transformées.

    Args:
        y_train (pandas.DataFrame): Le DataFrame contenant les données d'entraînement de la variable cible.
        y_test (pandas.DataFrame): Le DataFrame contenant les données de test de la variable cible.

    Returns:
        pandas.DataFrame: Un nouveau DataFrame avec les données numériques centrées et réduites.
    """

	# Application de la méthode StandardScaler aux variables numériques
    scaler = StandardScaler()
    y_train_scaller = pd.DataFrame(scaler.fit_transform(y_train),columns= y_train.columns)
    y_test_scaller = pd.DataFrame(scaler.transform(y_test), columns= y_test.columns)

    return y_train_scaller, y_test_scaller
