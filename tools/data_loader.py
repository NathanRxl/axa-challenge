import pandas as pd


def data_loader(path, nrows):
    # load row dataset from csv file
    row_dataset = pd.read_csv(path, nrows=nrows, sep=";")

    # TODO: retrieve X_train, y_train and X_test from row_dataset
    X_train = 0
    X_test = 0
    y_train = 0

    return X_train, y_train, X_test
