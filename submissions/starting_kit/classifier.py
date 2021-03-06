import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as RFC

class Classifier(BaseEstimator):
    def __init__(self, clf=None):
        self.clf = RFC(n_estimators=14, max_depth=18, max_features=150)
        if clf is not None:
            self.clf = clf

    def fit(self, X, y):
        # y = y.fillna(-1)
        y_reduced_format = y[y != -1]
        self.clf.fit(X, y_reduced_format)

    def predict_proba(self, X):
        n_lines = X.n_lines.values
        preds_reduced_format = self.clf.predict_proba(X)
        preds_reduced_format = np.hstack([np.zeros((preds_reduced_format.shape[0], 1)), preds_reduced_format])

        n_classes = preds_reduced_format.shape[1]
        preds_extended_format = np.zeros((n_lines.sum(), n_classes)) + 1 / n_classes
        idx = np.cumsum(n_lines)
        preds_extended_format[idx - 1] = preds_reduced_format
        
        return preds_extended_format

    def predict(self, X):
        n_lines = X.n_lines.values
        preds_reduced_format = self.clf.predict(X)

        preds_extended_format = np.zeros(n_lines.sum()) - 1
        idx = np.cumsum(n_lines)
        preds_extended_format[idx - 1] = preds_reduced_format
        
        return preds_extended_format
