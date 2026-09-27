import os

import numpy as np
from utils.spot import SPOT
# from src.constants import lm  #  lm = [0.02, 3]
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from metrics.get_all_evaluation_score import get_all_evaluation_score

class POTThresholdDetector:
    def __init__(self, q=1e-5, level=0.02, scale_factor=3.0):
        """
        (SPOT) 

        Args:
            q (float): Risk parameter controlling sensitivity (default 1e-5)
            level (float): Initial threshold quantile (default 0.02)
            scale_factor (float): Scaling factor for the final threshold (default 3.0)
        """
        self.q = q
        self.level = level
        self.scale_factor = scale_factor
        self.spot = None
        self.threshold = None

    def fit(self, train_scores, test_scores):
        """
        Train the model and compute the dynamic threshold

        Args:
            train_scores (np.ndarray): Training set anomaly scores (used to initialize SPOT)
            test_scores (np.ndarray): Test set anomaly scores (used for detection)
        """
        # Dynamically adjust the initial quantile until successful
        current_level = self.level
        while True:
            try:
                self.spot = SPOT(self.q)
                self.spot.fit(train_scores, test_scores)
                self.spot.initialize(level=current_level, min_extrema=False, verbose=False)
                break
            except:
                current_level *= 0.98  # Gradually decrease the quantile

        # Run SPOT to obtain the threshold
        results = self.spot.run(dynamic=False)

        # Compute the final threshold
        if len(results['thresholds']) > 0:
            self.threshold = np.mean(results['thresholds']) * self.scale_factor
        else:
            self.threshold = np.percentile(test_scores, 100 * (1 - self.level)) * self.scale_factor

    def predict(self, scores):
        """
        Generate predicted labels based on the threshold

        Args:
            scores (np.ndarray): Anomaly scores to be predicted

        Returns:
            np.ndarray: Binary predicted labels (0/1)
        """
        return (scores > self.threshold).astype(int)

    def evaluate(self, scores, labels):
        """
        Evaluate the threshold performance

        Args:
            scores (np.ndarray): Test set anomaly scores
            labels (np.ndarray): Ground truth labels (0/1)

        Returns:
            dict: Metrics including F1, Precision, Recall, AUC, etc.
        """
        pred = self.predict(scores)
        res = get_all_evaluation_score(pred, labels)

        res = {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in res.items()
        }

        return res, self.threshold