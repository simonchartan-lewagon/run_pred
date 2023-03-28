# Importation des librairies nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Standard
def scaler_standard(train, test):
    train_float = train.select_dtypes(include='float')
    test_float = test.select_dtypes(include='float')
    train_ex = train.select_dtypes(exclude='float').reset_index()
    test_ex = test.select_dtypes(exclude='float').reset_index()

    scaler = StandardScaler()
    train_float_scaller = pd.DataFrame(scaler.fit_transform(train_float),columns= train_float.columns)
    test_float_scaller = pd.DataFrame(scaler.transform(test_float), columns= test_float.columns)

    train_scaller = pd.concat([train_ex, train_float_scaller],axis=1)
    test_scaller =  pd.concat([test_ex, test_float_scaller])
    train_scaller = train_scaller.drop(columns='index')
    test_scaller = test_scaller.drop(columns='index')

    return train_scaller, test_scaller
