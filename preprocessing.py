import json

import pandas as pd

from sklearn.model_selection import train_test_split

class DataParser():

    def __init__(self, filepaths: tuple):
        '''
        Constructor of class DataParser
        '''
        self.filepaths = filepaths
    
    def parse_data(self) -> tuple:
        '''
        Parses the data from the given filepaths and returns the data in the form of a tuple.
        '''
        data = []
        for filepath in self.filepaths:
            with open(filepath, 'r') as f:
                data.append(json.load(f))
        return list(data)
    
    def print_data(self, data: list, all: bool=True) -> None:
        '''
        Prints the data in the form of a tuple.
        '''
        if all:
            print(data)
        else:
            print(f'Example of text:\n\n{data[0]["full_text"]}')
            
        return 'Success'
    
    def preprocess(self, data: list) -> pd.DataFrame:
        '''
        Performs preprocessing on the data.
        '''
        # TODO realize preprocessing for texts
        final_df = []
        for index, items in enumerate(data):
            _, _, tokens, trailing_whitespace, labels =   data[index]['document'], data[index]['full_text'], data[index]['tokens'], data[index]['trailing_whitespace'], data[index]['labels']
            dataframe = pd.DataFrame({'tokens': tokens,
                                      'trailing_whitespace': trailing_whitespace,
                                      'labels': labels})
            final_df.append(dataframe)
        
        result_df = pd.concat(final_df).drop_duplicates(subset=['tokens']).reset_index(drop=True)

        X, y = result_df.drop('labels', axis=1), result_df['labels']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return X_train, X_test, y_train, y_test
