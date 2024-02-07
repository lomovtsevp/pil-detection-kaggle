import json

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
    
    def print_data(self, data: list, all=True):
        '''
        Prints the data in the form of a tuple.
        '''
        if all:
            print(data)
        else:
            print(f'Example of text:\n\n{data[0]["full_text"]}')
            
        return 'Success'
    

if __name__ == "__main__":
    parser = DataParser(filepaths=('data/train.json', 'data/test.json'))
    train, test = parser.parse_data()
    parser.print_data(train)



