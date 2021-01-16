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
    df['all'] = df[0]
    df = df.drop(columns=[0, 1]) # drop car names

    X, y = np.zeros((398, 7)), np.zeros(398)
    for i, row in df.iterrows():
        split = row[0].split()
        if '?' not in split:
            split = np.array([float(num) for num in split])
            y[i] = split[0]
            X[i] = split[1:]

    save_to_csv('auto_mpg', X, y)


if __name__ == "__main__":
    os.makedirs('data/', exist_ok=True)

    get_auto_mpg()