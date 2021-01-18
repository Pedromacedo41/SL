import pandas as pd


def read(name):
    X = pd.read_csv('data/X_{}.csv'.format(name), header=None)
    y = pd.read_csv('data/y_{}.csv'.format(name), header=None)
    X['y'] = y[0]
    return X


def save(name, df):
    y = df['y']
    X = df.drop(columns=['y'])
    X.to_csv('data/X_{}.csv'.format(name), header=None, index=None)
    y.to_csv('data/y_{}.csv'.format(name), header=None, index=None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    df = read(args.name)
    df = df.sample(frac=1.)
    save(args.name, df)
