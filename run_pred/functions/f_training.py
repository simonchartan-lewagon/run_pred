import pandas as pd
import numpy as np
import time
import os
import joblib

from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data
from run_pred.functions.c_feature_engineering import engineer_features
from run_pred.functions.d_balancing import balance_data
from run_pred.functions.e_scaling_encoding import scale_encode_data

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor

from google.cloud import storage

# Note: the best model has been searched and engineered into individual notebooks,
# by each member of the team. Only the best model is train and stored here, see below.

def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series) -> object:

    """
    Train a stacking regression model on the input training dataset `X_train` and `y_train`,
    and evaluate its performance on the test dataset `X_test` and `y_test`.
    The function defines the hyperparameters and models for three base regression models:
    Random Forest Regressor, XGBoost Regressor, and Gradient Boosting Regressor.

    It then trains a meta-regressor, which is a Linear Regression model,
    using the predicted outputs of the three base models as its input.

    Finally, it constructs a Stacking Regressor, which combines the three base models
    and the meta-regressor, and trains it on the input training dataset.

    Args:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data target.

    Returns:
    - model (object): Trained StackingRegressor model.
    """

    # Defining the parameters for each sub-model
    xgb_params = {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 25, 'min_child_weight': 1, 'n_estimators': 1200}
    gb_params = {'loss': 'huber', 'max_depth': 15, 'max_features': 1.0, 'min_samples_split': 6, 'n_estimators': 500}

    # Defining the models
    rf_model = RandomForestRegressor()
    xgb_model = XGBRegressor(**xgb_params)
    gb_model = GradientBoostingRegressor(**gb_params)

    # Defining the meta-regressor
    meta_regressor = LinearRegression()

    # Defining the StackingRegressor by combining the sub_models defined above
    model = StackingRegressor(
        estimators=[ ('gb', gb_model),('xgb', xgb_model),('rf', rf_model)],
        final_estimator=meta_regressor)

    # Training the model
    model.fit(X_train,y_train)

    # Checking the model results
    print(f'R² = {model.score(X_test,y_test).round(3)}')
    print(f'MAPE = {mean_absolute_percentage_error(y_test, model.predict(X_test)).round(3)}')

    return model


def save_model(model: object) -> None:
    """
    Saves the trained model on hard drive at f"{models/{model_name}.joblib"
    """
    if not os.path.isdir('models'): os.mkdir('models')

    model_name = type(model).__name__
    print(f'Saving {model_name} model...')

    # Saving model locally
    model_path = os.path.join("models", f"{model_name}.joblib")
    joblib.dump(model, open(model_path, 'wb'))
    print(f'{model_name} model saved')


def load_model(model: str) -> object:
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
        # ID of the GCS bucket
        bucket_name = "run_pred_model"

        # ID of the GCS object
        source_blob_name = model_path

        # Path to which the file should be downloaded
        destination_file_name = model_path

        storage_client = storage.Client.from_service_account_json('gcs_credentials.json')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f'{model} model downloaded from GCS')

    model = joblib.load(model_path)
    return model


if __name__ == '__main__' :

    # Testing the data whole preprocessing
    ## dataset = clean_data('raw_data/raw-data-kaggle.csv')
    ## X_train_raw, X_test_raw, y_train, y_test = split_data(dataset)
    ## X_train_feat = engineer_features(X_train_raw, y_train)
    ## X_test_feat = engineer_features(X_test_raw, y_test)
    ## X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train)
    ## X_train_balanced_scaled_encoded, X_test_scaled_encoded = scale_encode_data(X_train_balanced, X_test_feat)

    # Testing the load_model and save_model functions
    ## model = load_model('StandardScaler')
    ## assert(model is not None)
    ## print(type(model))
    ## save_model(model)

    # Training the best model on the train dataset (28k rows), evaluating its performance, and saving the model
    ## dataset = clean_data('raw_data/raw-data-kaggle.csv')
    ## X_train_raw, X_test_raw, y_train, y_test = split_data(dataset)
    ## X_train_feat = engineer_features(X_train_raw, y_train)
    ## X_test_feat = engineer_features(X_test_raw, y_test)
    ## X_train_balanced, y_train_balanced = balance_data(X_train_feat=X_train_feat, y_train=y_train)
    ## X_train_balanced_scaled_encoded, X_test_scaled_encoded = scale_encode_data(X_train_balanced, X_test_feat)
    ## model = train_best_model(X_train_balanced_scaled_encoded, y_train_balanced, X_test_scaled_encoded, y_test)
    ## save_model(model)

    # Training the validated best model on the FULL dataset (32k rows) and saving it
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    print(dataset.head())
    X = dataset.drop(columns = ['time'])
    y = dataset.time
    X_feat = engineer_features(X, y)
    X_balanced, y_balanced = balance_data(X_train_feat=X_feat, y_train=y)
    X_balanced_scaled_encoded = scale_encode_data(X_balanced, X_balanced, save=True)[0]
    model = train_best_model(X_balanced_scaled_encoded, y_balanced, X_balanced_scaled_encoded, y_balanced)
    save_model(model)
