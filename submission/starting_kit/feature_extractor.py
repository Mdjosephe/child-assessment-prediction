import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import json


import os
import pandas as pd
import numpy as np






class FeatureExtractor:

    def __init__(self):
        self.train_cols = None

    def fit(self, X, y):

        numeric_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
        num_cols =[i for i in X.columns][:-1]
        drop_cols = 'installation_id'

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('drop cols', 'drop', drop_cols),
                ])

        self.preprocessor = preprocessor
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)
