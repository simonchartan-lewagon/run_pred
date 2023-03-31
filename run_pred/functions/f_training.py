import pandas as pd
import numpy as np
import time
import glob
import os
import joblib

from run_pred.params import *
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data
from run_pred.functions.c_feature_engineering import engineer_features
from run_pred.functions.d_balancing import balance_data
from run_pred.functions.e_scaling_encoding import scale_encode_data

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor

# Note: the best model has been searched and engineered into individual notebooks,
# by each member of the team. Only the best model is train and stored here, see below.

def train_best_model(X_train, y_train, X_test, y_test):

    # Définir les paramètres pour chaque modèle de régression
    xgb_params = {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 25, 'min_child_weight': 1, 'n_estimators': 1200}
    gb_params = {'loss': 'huber', 'max_depth': 15, 'max_features': 1.0, 'min_samples_split': 6, 'n_estimators': 500}

    # Définir les modèles de régression
    rf_model = RandomForestRegressor()
    xgb_model = XGBRegressor(**xgb_params)
    gb_model = GradientBoostingRegressor(**gb_params)

    # Définir le modèle meta-régresseur
    meta_regressor = LinearRegression()

    #memo cked_model = StackingRegressor(estimators=[('rf', rf_model), ('xgb', xgb_model), ('gb', gb_model)],
    # Définir le modèle de Stacking en utilisant les modèles de régression de base et le modèle meta-régresseur
    model = StackingRegressor(
        estimators=[ ('gb', gb_model),('xgb', xgb_model),('rf', rf_model)],
        final_estimator=meta_regressor)

    # Train the model
    model.fit(X_train,y_train)

    # Check the model results
    print(f'R² = {model.score(X_test,y_test).round(3)}')
    print(f'MAPE = {mean_absolute_percentage_error(y_test, model.predict(X_test)).round(3)}')

    return model


def save_model(model):
    """
    Saves the model trained on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{model_name}_{timestamp}.joblib"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = type(model).__name__

    # save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{model_name}_{timestamp}.joblib")
    joblib.dump(model, open(model_path, 'wb'))


def load_model():
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "RandomForestRegressor_20230331-172305.joblib")
    model = joblib.load(model_path)
    return model

if __name__ == '__main__' :
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    X_train_raw, X_test_raw, y_train, y_test = split_data(dataset)
    X_train_feat = engineer_features(X_train_raw, y_train)
    X_test_feat = engineer_features(X_test_raw, y_test)
    X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train)
    X_train_balanced_scaled_encoded, X_test_scaled_encoded = scale_encode_data(X_train_balanced, X_test_feat)
    model = train_best_model(X_train_balanced_scaled_encoded, y_train_balanced, X_test_scaled_encoded, y_test)
    save_model(model)
    #model = load_model()
