import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class Model():

    def __init__(self, params: dict=None):
        
        # TODO Import roBERTa model for Token Classification.

        if params:
            self.model = XGBClassifier(random_state=42, params=params)
        else:
            self.model = XGBClassifier(random_state=42, enable_categorical=True)

    def train(self, data: pd.DataFrame, target: np.array) -> None:
        '''
        Trains the model.
        '''
        target = LabelEncoder().fit_transform(target)
        self.model.fit(data, target)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Predicts the data.
        '''
        return self.model.predict(data)
