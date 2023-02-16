import joblib
import os
import pandas as pd
import numpy as np
from config import EXPERIMENT_NAME, MODEL_DIR, SCALER_DIR
from .model import root_mean_square_error
from config import MODEL, TARGET_INDEX
import keras

# def get_model_name(model_index):
#     return 'model_' + model_index + '.pkl'

def load_model(model_dir, model_name, is_scaler):
    current_directory = os.getcwd()
    model_filename = os.path.join(current_directory, model_dir, EXPERIMENT_NAME, model_name)
    if(MODEL=='CNN' and not is_scaler):
        model = keras.models.load_model(model_filename)
    else:
        model = joblib.load(model_filename)

    return model

def predict_one_step(model, X):
    # model = load_model(model_index, experiment_name)
    if(MODEL=='VAR'):
        pred = model.forecast(y=X[-1:], steps=1)
        pred = pred[:, pred.shape[1] - 1]
    elif(MODEL=='CNN'):
        pred = model.predict(X)
        pred = pred[:, 0]
    else:
        pred = model.predict(X)
    return pred[0]

# Finds corresponding subsets for model with model_index and scales it with a corresponding scaler
def get_subset_X_for_one(X, sample_subsets, model_index):
    features_index = sample_subsets[int(model_index) - 1]
    subset_X = X.iloc[features_index]
    subset_X = subset_X.values.reshape(1, -1)

    scaler_name = 'X_scaler_' + model_index + '.pkl'
    scaler = load_model(SCALER_DIR, scaler_name, True)
    subset_X = scaler.transform(subset_X)
    if(MODEL == 'VAR'):
        # subset_X = subset_X.to_numpy()
        # subset_X = np.reshape(subset_X, (1, -1))
        y = X.iloc[TARGET_INDEX]
        y = np.reshape(y, (-1, 1))
        subset_X = np.concatenate((subset_X, y), axis=1)
    return subset_X

# Finds corresponding subsets for model with model_index and scales it with a corresponding scaler
def get_subset_X_for_n(X, sample_subsets, model_index):
    features_index = sample_subsets[int(model_index) - 1]
    subset_X = X.iloc[:, features_index]

    scaler_name = 'X_scaler_' + model_index + '.pkl'
    scaler = load_model(SCALER_DIR, scaler_name, True)
    subset_X = scaler.transform(subset_X)
    if(MODEL == 'VAR'):
        y = X.iloc[:, TARGET_INDEX]
        y = y.to_numpy()
        y = np.reshape(y, (-1, 1))
        subset_X = np.concatenate((subset_X, y), axis=1)
    return subset_X

def predict_one_step_for_ensemble(model_indices, X_all, sample_subsets):
    preds = []
    for model_index in model_indices:
        X = get_subset_X_for_one(X_all, sample_subsets, model_index)
        if(MODEL=='CNN'):
            model_name = 'model_' + model_index + '.h5'
        else:
            model_name = 'model_' + model_index + '.pkl'
        model = load_model(MODEL_DIR, model_name, False)
        pred = predict_one_step(model, X)
        preds.append(pred)
    return preds

# len of X should be > 1
def predict_n_steps_for_ensemble(model_indices, X_all, sample_subsets):
    n_steps_pred = []
    for model_index in model_indices:
        X = get_subset_X_for_n(X_all, sample_subsets, model_index)
        if(MODEL=='CNN'):
            model_name = 'model_' + model_index + '.h5'
        else:
            model_name = 'model_' + model_index + '.pkl'
        model = load_model(MODEL_DIR, model_name, False)
        if(MODEL=='VAR'):
            pred = model.forecast(y=X[-1:], steps=X.shape[0])
            pred = pred[:, pred.shape[1] - 1]
        elif(MODEL=='CNN'):
            pred = model.predict(X)
            pred = pred[:, 0]
        else:
            pred = model.predict(X)
        n_steps_pred.append(pred)
    # for _, row in X_all.iterrows():
        # one_step_pred = predict_one_step_for_ensemble(model_indices, row, sample_subsets)
        # n_steps_pred.append(one_step_pred)
    return n_steps_pred

def get_weights(ensemble, X, y, sample_subsets):
    preds = predict_n_steps_for_ensemble(ensemble, X, sample_subsets)
    preds = pd.DataFrame(preds)
    errors = []
    for _, col in preds.iterrows():
        error = root_mean_square_error(y.to_numpy(), col.to_numpy())
        errors.append(error)
    weights = [1/x for x in errors]
    return weights

def final_prediction_ensemble(ensemble, X, sample_subsets, weights):
    pred = predict_one_step_for_ensemble(ensemble, X, sample_subsets)
    numerator = sum([w*x for w,x in zip(weights, pred)])
    denominator = sum(weights)
    weighted_average = numerator/denominator
    return weighted_average