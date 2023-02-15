from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series
import numpy as np
import pandas as pd
import os
import joblib
from joblib import Parallel, delayed
from config import MODEL_DIR, EXPERIMENT_DIR, EXPERIMENT_NAME, SCALER_DIR
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, MaxPooling1D

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

def train_candidates(train, val, target_index, sample_subsets, model_name):

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

        if(model_name == 'CNN'):
            model = train_cnn(X_train, y_train)
            prediction = model.predict(X_val)
        elif(model_name == 'DT'):
            model = DecisionTreeRegressor(max_depth=2, random_state=0)
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
        elif(model_name == 'VAR'):
            var_y_train = y_train.to_numpy()
            var_y_train = np.reshape(var_y_train, (-1, 1))
            var_train = np.concatenate((X_train, var_y_train), axis=1)
            var = VAR(var_train)
            model = var.fit(1)
            val_horizon = y_val.shape[0]
            prediction = model.forecast(y=var_train[-1:], steps=val_horizon)
            prediction = prediction[:, prediction.shape[1] - 1]
        else:
            model = None

        val_rmse = root_mean_square_error(y_val, prediction.flatten())

        print(f'Validation error for model {counter} is {val_rmse}.')
        predictions.append(prediction)
        save_model(model, model_dir, 'model', counter)

        counter += 1

    val_predictions = pd.DataFrame(predictions)
    val_predictions = val_predictions.transpose()

    save_validation_predictions(val_predictions, 'init')

def train_cnn(X_train, y_train):
        model = Sequential()
        model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=1))
        model.add(LSTM(10, ))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30)

        return model
def save_validation_predictions(predictions, version):
    n = len(predictions.columns) + 1
    predictions.columns = range(1, n)

    experiment_dir = create_directory(EXPERIMENT_DIR)
    # filename = experiment_dir + f'\\validation_predictions_{version}.csv'
    filename = experiment_dir + f'/validation_predictions_{version}.csv'
    predictions.to_csv(filename, encoding='utf-8', index=False)

def get_avg_silhoutte(predictions, num_clusters):
    X = to_time_series(predictions)[:, :, np.newaxis]

    # for num_clusters in num_range:
        
        # initialise kmeans
    kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="softdtw")
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
        
    # silhouette score
    return (num_clusters, silhouette_score(X, cluster_labels))

# Data has to be in the shape of (num_ts, size_ts)
def get_best_num_of_clusters(data, num_range=range(2, 15)):
    results = Parallel(n_jobs=10)(delayed(get_avg_silhoutte)(data, i) for i in num_range)
    avg_silhouttes = np.array(results)
    max_avg_silhoutte = (avg_silhouttes[:, 1] == max(avg_silhouttes[:, 1]))

    best_num_clusters = avg_silhouttes[max_avg_silhoutte][0][0]
    print(f'Determined number of clusters: {best_num_clusters}')

    return int(best_num_clusters)

# Data has to be in the shape of (num_ts, size_ts)
def cluster_predictions(predictions, num_clusters):
    # num_clusters=int(np.max(DBSCAN(eps=0.05,min_samples=2).fit(predictions.transpose()).labels_))+1
    X = to_time_series(predictions)[:, :, np.newaxis]
    kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="softdtw", max_iter=10)

    # kmeans = KMeans(
    #     init="random",
    #     n_clusters=num_clusters,
    #     n_init=10,
    #     max_iter=300,
    #     random_state=42
    # )
    kmeans.fit(X)

    return kmeans.labels_, kmeans.cluster_centers_[:, :, 0]

def compute_cluster_representatives(labels, cluster_centers, predictions):
    ensemble = []
    # Should be automatically tuned
    n_clusters = cluster_centers.shape[0]
    for i in range(n_clusters):
        i_center = cluster_centers[i, ]
        i_data = predictions.loc[:, labels==i]
        distance_sum = np.abs(np.sum(i_data.transpose() - i_center, axis=1))
        if(len(distance_sum) != 0):
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