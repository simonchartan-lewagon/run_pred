#Installation de la librairie imbalanced-learn
#! pip install -U imbalanced-learn

# Importation des librairies nécessaires
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def balancing_gender(X_train, y_train):
    """
    Cette fonction utilise la technique de suréchantillonnage (oversampling) pour équilibrer le nombre d'échantillons
    pour chaque classe de la variable cible "gender" dans un DataFrame. Elle prend en entrée les données d'apprentissage
    déjà séparées en features et variable cible, effectue le suréchantillonnage et retourne les nouvelles données
    d'apprentissage.

    Args:
        X_train (pandas.DataFrame): Le DataFrame contenant les features pour l'ensemble d'apprentissage.
        y_train (pandas.Series): La Series contenant la variable cible pour l'ensemble d'apprentissage.

    """

    # Concaténation de X_train et y_train
    train = X_train
    train['time'] = y_train
    train.head(3)

    # Définition des features et de la variable cible pour le suréchantillonnage
    X_bal = train.drop('gender', axis=1)
    y_bal = train['gender']

    #  Instanciation de RandomOverSampler et duplication des lignes de la classe sous-représentée
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X_bal, y_bal)

    # Concaténation de X_resampled et y_resampled
    dfb_train = X_resampled
    dfb_train['gender']= y_resampled

    # remise de la target
    X_train_ball = dfb_train.drop('time', axis=1)
    y_train_ball = dfb_train['time']


    return X_train_ball, y_train_ball
