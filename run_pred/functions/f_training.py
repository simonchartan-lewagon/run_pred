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

from google.cloud import storage

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
    Saves the model trained on hard drive at f"{models/{model_name}_{timestamp}.joblib"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = type(model).__name__

    # save model locally
    model_path = os.path.join("models", f"{model_name}_{timestamp}.joblib")
    joblib.dump(model, open(model_path, 'wb'))


def load_model(model):
    """
    This function loads a given model, which can be one of these:
        - StandardScaler
        - OneHotEncoder
        - StackingRegressor
    """
    if not os.path.isdir('models'): os.mkdir('models')

    model_path = os.path.join("models", f"{model}.joblib")

    if not os.path.isfile(model_path):
        print(f'Downloading {model} model from GCS...')
        # The ID of your GCS bucket
        bucket_name = "run_pred_model"

        # The ID of your GCS object
        source_blob_name = model_path

        # The path to which the file should be downloaded
        destination_file_name = model_path

        storage_client = storage.Client.from_service_account_json('gcs_credentials.json')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f'{model} model downloaded from GCS')

    model = joblib.load(model_path)
    return model

if __name__ == '__main__' :
    #dataset = clean_data('raw_data/raw-data-kaggle.csv')
    #X_train_raw, X_test_raw, y_train, y_test = split_data(dataset)
    #X_train_feat = engineer_features(X_train_raw, y_train)
    #X_test_feat = engineer_features(X_test_raw, y_test)
    #X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train)
    #X_train_balanced_scaled_encoded, X_test_scaled_encoded = scale_encode_data(X_train_balanced, X_test_feat)
    #model = train_best_model(X_train_balanced_scaled_encoded, y_train_balanced, X_test_scaled_encoded, y_test)
    #save_model(model)
    #model = load_model()

    model = load_model('StandardScaler')
    assert(model is not None)
    print(type(model))
