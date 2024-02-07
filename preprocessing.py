import json
import pandas as pd

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
        return tuple(data)
    
    def print_data(self, data: list, all: bool=True) -> None:
        '''
        Prints the data in the form of a tuple.
        '''
        if all:
            print(data)
        else:
            print(f'Example of text:\n\n{data[0]["full_text"]}')
            
        return 'Success'
    
    def preprocessing(self, data: list) -> list:
        '''
        Performs preprocessing on the data.
        '''
        # TODO realize preprocessing for texts

        for index, items in enumerate(data):
            text, tokens, trailing_whitespace, labels = data[index]['full_text'], data[index]['tokens'], data[index]['trailing_whitespace'], data[index]['labels']
            dataframe = pd.DataFrame({'text': text, 'tokens': tokens, 'trailing_whitespace': trailing_whitespace, 'labels': labels})
            print(dataframe['labels'].unique())
            break
        return [0]
    

if __name__ == "__main__":
    parser = DataParser(filepaths=('data/train.json', 'data/test.json'))
    train, test = parser.parse_data()
    parser.print_data(train, all=False)
    parser.preprocessing(train)



