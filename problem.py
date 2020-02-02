import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit
import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType

problem_title = 'Children Assessment Perfomance Prediction'
_target_column_name = 'accuracy_group'
_prediction_label_names = [-1, 0, 1, 2, 3]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
# An object implementing the workflow
workflow = FeatureExtractorClassifier()

# define the score (specific score for the FAN problem)
class DataScienceBowlScore(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='DSB score', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if len(y_true.shape) == 1:
            last_rows = y_true != -1
            y_pred = y_pred[last_rows]
            y_true = y_true[last_rows]
        else:
            last_rows = y_true[:, 0] == 0
            y_pred = y_pred[last_rows, 1:]
            y_true = y_true[last_rows, 1:]
            y_pred = np.array(_prediction_label_names)[np.argmax(y_pred, axis=1)]
            y_true = np.array(_prediction_label_names)[np.argmax(y_true, axis=1)]

        score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        return score

score_types = [DataScienceBowlScore()]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X['installation_id'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,
                       compression='zip')
    y_array = data[_target_column_name].fillna(-1).values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'train.csv.zip'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv.zip'
    return _read_data(path, f_name)