MODEL_DIR = 'models'
EXPERIMENT_DIR = 'experiments'
SCALER_DIR = 'scalers'

# Changes for each experiment, should be the name of the dataset folder
EXPERIMENT_NAME = 'sml10-dataset'
# Size of validation window for clustering
VALIDATION_WINDOW_SIZE = 50
# Size of window on which compute predictions to define weights
WEIGHTS_WINDOW_SIZE = 8
# Window for drifts evaluation
EVALUATION_WINDOW = 20

# Hyperparameters

#Target column
TARGET_INDEX = 2
# Size of feature subset
SUBSET_SIZE = 0.25
# Number of randomly selected candidates
K = 100