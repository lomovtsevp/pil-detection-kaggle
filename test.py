import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report)
import seqeval


class Validator():

    def __init__(self, model: object, X_test: pd.DataFrame, y_test: np.array):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, metric: object) -> float:
        '''
        Evaluates model by calculating given metric.
        '''
        predictions = self.model.predict(self.X_test)

        if self.y_test.nunique() > 2: #for multiclass
            return metric(self.y_test, predictions, average='weighted')
        return metric(self.y_test, predictions)
    
    def get_report(self) -> str:
        '''
        Calculates report based on sklearn.
        '''
        predictions = self.model.predict(self.X_test)

        if self.y_test.nunique() > 2: #for multiclass
            return classification_report(self.y_test, predictions, average='weighted', target_names=self.y_test.unique())
        return classification_report(self.y_test, self.model.predict(self.X_test), target_names=self.y_test.unique())