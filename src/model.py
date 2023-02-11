from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import joblib
from joblib import Parallel, delayed
from config import MODEL_DIR, EXPERIMENT_DIR, EXPERIMENT_NAME, SCALER_DIR
import matplotlib.pyplot as plt

def root_mean_square_error(y_test, y_predicted):
    return np.mean(np.square(((y_test - y_predicted) / y_test)), axis=0)

def rmspe(y_test, y_predicted):
    return np.sqrt(root_mean_square_error(y_test, y_predicted))

# Creates directory cwd/main_dir/experiment_name if it does not exist
def create_directory(main_dir):
    current_directory = os.getcwd()

    directory = os.path.join(current_directory, main_dir, EXPERIMENT_NAME)
    CHECK_FOLDER = os.path.isdir(directory)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(directory)
        print("Created directory : ", directory)

    return directory

def save_model(model, model_dir, model_name, counter):
    model_location = os.path.join(model_dir, f'{model_name}_{counter}.pkl')
    joblib.dump(model, model_location)

def train_candidates(train, val, target_index, sample_subsets):

    model_dir = create_directory(MODEL_DIR)
    scaler_dir = create_directory(SCALER_DIR)

    y_train = train.iloc[:, target_index]
    y_val = val.iloc[:, target_index]

    # y_scaler = MinMaxScaler()
    # y_train = y_scaler.fit_transform(y_train)
    # y_val = y_scaler.transform(y_val)
    # save_model(y_scaler, scaler_dir, 'y_scaler', None)

    predictions = []
    counter = 1
    for subset in sample_subsets:
        X_train = train.iloc[:, subset]
        X_val = val.iloc[:, subset]

        X_scaler = MinMaxScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_val = X_scaler.transform(X_val)
        save_model(X_scaler, scaler_dir, 'X_scaler', counter)

        model = RandomForestRegressor(max_depth=2, random_state=0)
        model.fit(X_train, y_train)

        prediction = model.predict(X_val)
        val_rmse = root_mean_square_error(y_val, prediction)

        print(f'Validation error for model {counter} is {val_rmse}.')
        predictions.append(prediction)
        save_model(model, model_dir, 'model', counter)

        counter += 1

    val_predictions = pd.DataFrame(predictions)
    val_predictions = val_predictions.transpose()

    save_validation_predictions(val_predictions, 'init')

def save_validation_predictions(predictions, version):
    n = len(predictions.columns) + 1
    predictions.columns = range(1, n)

    experiment_dir = create_directory(EXPERIMENT_DIR)
    # filename = experiment_dir + f'\\validation_predictions_{version}.csv'
    filename = experiment_dir + f'/validation_predictions_{version}.csv'
    predictions.to_csv(filename, encoding='utf-8', index=False)

def cluster_predictions(predictions):
    
    num_clusters=int(np.max(DBSCAN(eps=0.05,min_samples=2).fit(predictions.transpose()).labels_))+1
    print(f'Determined number of clusters: {num_clusters}')
    kmeans = KMeans(
        init="random",
        n_clusters=num_clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(predictions.transpose())

    return kmeans.labels_, kmeans.cluster_centers_

def compute_cluster_representatives(labels, cluster_centers, predictions):
    ensemble = []
    # Should be automatically tuned
    n_clusters = cluster_centers.shape[0]
    for i in range(n_clusters):
        i_center = cluster_centers[i, ]
        i_data = predictions.loc[:, labels==i]
        distance_sum = np.abs(np.sum(i_data.transpose() - i_center, axis=1))
        representative_model_index = distance_sum.idxmin()
        ensemble.append(representative_model_index)
    
    return ensemble

def plot_clustering(labels, data):
    plt.rcParams["figure.figsize"] = (20,20)

    u_labels = np.unique(labels)
    
    #plotting the results:
    for i in u_labels:
        filtered_label = data.loc[:, labels == i]
        plt.plot(range(len(filtered_label)) , filtered_label, label=i)
    plt.legend(u_labels)
    return plt