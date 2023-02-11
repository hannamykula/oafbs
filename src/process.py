import pandas as pd
import numpy as np
from itertools import combinations, chain
import sys
import math
import logging
import random

DATA_DIR = '\data\\'

def read_and_split_data(filename, test_split = 0.8, val_size = 50):
    file_path = sys.path[0] + DATA_DIR + filename
    logging.info("Loading data from {}".format(filename))

    dataset = pd.read_csv(file_path)
    length = len(dataset)
    test_index = math.floor(length * test_split)
    val_index = test_index - val_size

    train = dataset[0:val_index]
    val = dataset[val_index:test_index]
    test = dataset[test_index:]

    assert length == len(train) + len(val) + len(test)
    return train, val, test

def subsample_features(dataset, target_index, subset_size = 0.1, k = 50):

    n_cols = len(dataset.columns)
    # We minus 1 because we always want to include target column
    # p = math.floor(n_cols * subset_size) - 1
    p = math.floor(n_cols * subset_size)
    # Generate all possible combinations of subsets of size p
    ran = chain(range(target_index), range(target_index + 1, n_cols))
    subsets = combinations(ran, p)
    # Randomly sample k samples from all possible combinations
    pool = tuple(subsets)
    n = len(pool)

    if k > n:
        raise Exception(f"The parameter k is too high. The number of max. possible combinations is {n}")

    indices = sorted(random.sample(range(n), k))
    sample_subsets = tuple(list(pool[i]) for i in indices)
    
    # Append target column index to each selected subset
    # sample_subsets = tuple(sample + (target_index,) for sample in sample_subsets)
    return sample_subsets
