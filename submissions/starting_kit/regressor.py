from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as RFC

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RFC(n_estimators=14, max_depth=18, max_features=150)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict_proba(self, X):
        return self.reg.predict_proba(X)

    def predict(self, X):
            return self.reg.predict(X)
