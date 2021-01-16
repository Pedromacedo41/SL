import os

import pandas as pd
import numpy as np


def save_to_csv(name, X, y):
    fileX = 'data/X_{}.csv'.format(name)
    pd.DataFrame(X).to_csv(fileX, header=None, index=None)
    filey = 'data/y_{}.csv'.format(name)
    pd.DataFrame(y).to_csv(filey, header=None, index=None)


def get_auto_mpg():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

    df = pd.read_csv(url, sep='\t', header=None)
    df = df.drop(columns=[1]) # drop car names

    X, y = np.zeros((398, 7)), np.zeros(398)
    for i, row in df.iterrows():
        split = row[0].split()
        if '?' not in split:
            split = np.array([float(num) for num in split])
            y[i] = split[0]
            X[i] = split[1:]

    save_to_csv('auto_mpg', X, y)


def get_housing():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

    df = pd.read_csv(url, header=None)

    X, y = np.zeros((506, 13)), np.zeros(506)
    for i, row in df.iterrows():
        split = np.array([float(num) for num in row[0].split()])
        y[i] = split[-1]
        X[i] = split[:-1]

    save_to_csv('housing', X, y)


def get_crime():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'

    df = pd.read_csv(url, header=None, na_values='?').drop(columns=[3]) # drop string column
    df = df.dropna(axis=1)

    X = df.to_numpy()
    y = X[:, -1]
    X = X[:, :-1]

    save_to_csv('crime', X, y)


def get_forest_fires():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'

    df = pd.read_csv(url)
    df = df.drop(columns=['month', 'day'])

    X = df.to_numpy()
    y = X[:, -1]
    X = X[:, :-1]

    save_to_csv('forestfires', X, y)


def get_wisconsin():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data'

    df = pd.read_csv(url, header=None, na_values='?')
    df = df.dropna(axis=1)
    df = df.drop(columns=[0])
    df[1] = (df[1] == 'R').astype('float32')

    X = df.to_numpy()
    X[:, [0, 1]] = X[:, [1, 0]]
    y = X[:, 0]
    X = X[:, 1:]

    save_to_csv('wisconsin', X, y)


if __name__ == "__main__":
    os.makedirs('data/', exist_ok=True)

    get_auto_mpg()
    get_housing()
    get_crime()
    get_forest_fires()
    get_wisconsin()