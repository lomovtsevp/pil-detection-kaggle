from preprocessing import DataParser
from model import Model
from test import Validator

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    
    parser = DataParser(filepaths=('data/train.json', 'data/test.json'))
    model = Model()

    train, test = parser.parse_data()

    X_train, X_test, y_train, y_test = parser.preprocess(train)

    model.train(X_train, y_train, X_test, y_test)

    #0.899 submission





