from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class PageHinkley(BaseDriftDetector):
    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.minimum = None
        self.reset()

    def reset(self):
        """ reset
        Resets the change detector parameters.
        """
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0
        self.minimum = 1.0

    def add_element(self, x):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.
        
        Notes
        -----
        After calling this method, to verify if change was detected, one 
        should call the super method detected_change, which returns True 
        if concept drift was detected and False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        self.sum = self.sum + (x - self.x_mean - self.delta)
        self.minimum = min(self.minimum, self.sum)

        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if self.sample_count < self.min_instances:
            return None

        if self.sum - self.minimum > self.threshold:
            self.in_concept_change = True

class DataDrift():
    def __init__(self, threshold):
        self.threshold = threshold
        self.data_drift = False
        self.sample_count = 0
        self.min = None
        self.min2 = None
        self.difference = None

    def init_min(self, correlation):
        if (self.sample_count == 1):
            # initialize min
            self.min = min(correlation)
    # X and y - on the evaluation window
    def add_element(self, X, y):
        
        correlation = X.apply(lambda x: abs(np.corrcoef(x, y)[0, 1]), axis=0)
        self.sample_count += 1

        self.init_min(correlation)
        self.data_drift = False

        self.min2 = min(correlation)
        # will not be true for sample count 1
        self.difference = abs(self.min - self.min2)
        if(self.difference > np.sqrt(np.log(1/self.threshold)/(2*self.sample_count))):
            self.data_drift = True
            self.min = self.min2
            self.sample_count = 0


    def detected_change(self):
        return self.data_drift



