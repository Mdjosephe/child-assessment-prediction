import os
import numpy as np
import pandas as pd
import rampwf as rw
import json
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import cohen_kappa_score as scoring_kappa
import imp


preprocessing_file = imp.load_source('preprocessing',"C:\\Users\\mohamed.abdel-wedoud\\Desktop\\guikho\\submissions\\starting_kit\\preprocessing.py")
preprocessing = preprocessing_file.preprocessing





problem_title = 'Children Assessment Perfomance Prediction'
_target_column_name = 'Groups {0,1,2,3}'
Predictions = rw.prediction_types.make_regression()


class FAN(FeatureExtractorRegressor):

    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor']):
        super(FAN, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names


workflow = FAN()


import numpy as np


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

# define the score (specific score for the FAN problem)
class FAN_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='quadratic kappa error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        score = quadratic_weighted_kappa(y_true,y_pred)

        return score


score_types = [
    FAN_error(name='quadratic kappa error', precision=2),
]


def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=4, test_size=0.20, random_state=42)
    return cv.split(X, y,groups=X['installation_id'])


def truncate_and_get_y(df):
    """
    Truncates the dataset randomly for each user, according to the challenge procedure
    Also returns y before the information is lost through the truncation
    """

    # Sort dataframe
    df = df.copy()
    df.timestamp = pd.to_datetime(df.timestamp)
    df.sort_values(['installation_id', 'timestamp'], inplace=True)

    # Filter dataframe
    df_attempts = df[(df.type == 'Assessment') & \
                    ((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)') |
                     (df.event_code == 4110) & (df.title == 'Bird Measurer (Assessment)'))].copy()
    df = df[df.installation_id.isin(df_attempts.installation_id.unique())].copy().reset_index(drop=True)

    # Compute y
    df_attempts['success'] = df_attempts.event_data.apply(lambda x: json.loads(x)['correct']) * 1
    passed = df_attempts.groupby('game_session').success.max().to_frame('passed')
    n_attempts = df_attempts.groupby('game_session').size().to_frame('n_attempts')
    accuracy = pd.Series(0, index=passed.index)
    accuracy.loc[(passed.passed == 1) & (n_attempts.n_attempts > 2)] = 1
    accuracy.loc[(passed.passed == 1) & (n_attempts.n_attempts == 2)] = 2
    accuracy.loc[(passed.passed == 1) & (n_attempts.n_attempts == 1)] = 3

    selected_sessions = df_attempts.groupby('installation_id').game_session\
        .apply(lambda obj: obj.loc[np.random.choice(obj.index)])
    y = pd.Series(accuracy.loc[selected_sessions].values, index=selected_sessions.index)

    # Truncate
    df['idx'] = df.index
    marked_start = df.game_session.isin(selected_sessions) \
        & df.idx.isin(df.groupby('game_session').idx.first())
    marked_start = marked_start[marked_start].index
    marked_end = df.idx.isin(df.groupby('installation_id').idx.last())
    marked_end = marked_end[marked_end].index
    df.drop('idx', axis=1, inplace=True)
    idx = list(range(marked_start[0] + 1))
    for start, end in zip(marked_start[1:], marked_end[:-1]):
        idx += list(range(end + 1, start + 1))
    df = df.loc[idx]

    return preprocessing(df), y

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,compression='zip')
    X_df, y_array = truncate_and_get_y(data)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[::10], y_array[::10]
    else:
        return X_df, y_array
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.zip'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.zip'
    return _read_data(path, f_name)
