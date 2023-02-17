import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def select_relevant(X_train, y_train, X_test, k):
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    correlation = X_train.apply(lambda x: abs(np.corrcoef(x, y_train)[0, 1]), axis=0)
    top_k = correlation.sort_values(ascending=False)[0:k].index.values
    relevant_X_train = X_train.loc[:, top_k]
    relevant_X_test = X_test.loc[:, top_k]

    # relevant_X_scaler = MinMaxScaler()
    # relevant_X_train_scaled = relevant_X_scaler.fit_transform(relevant_X_train)
    # relevant_X_test_scaled = relevant_X_scaler.fit_transform(relevant_X_test)
    return relevant_X_train, relevant_X_test
