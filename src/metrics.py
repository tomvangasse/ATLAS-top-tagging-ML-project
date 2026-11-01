import numpy as np
from sklearn.metrics import roc_curve

def background_rejection(labels, scores, signal_efficiency, weights=None):
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)  # tpr/fpr is the true/false positive rate
    return 1 / np.interp(signal_efficiency, tpr, fpr)
