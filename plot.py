from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')


def get_logs(name, run):
    bart_csv = pd.read_csv('logs/bart_{}{}.csv'.format(name, run))
    bart_data = {
        'train_rmse': bart_csv['train_rmse'].item(),
        'val_rmse': bart_csv['val_rmse'].item(),
        'test_rmse': bart_csv['test_rmse'].item(),
    }

    with open('logs/nbart_{}{}/metrics.pkl'.format(name, run), 'rb') as f:
        metrics = pickle.load(f)

    nbart_data = {}
    for k in metrics.keys():
        k_arr = np.zeros(len(metrics[k]))
        for step, value in metrics[k]:
            k_arr[step-1] = value
        nbart_data[k] = k_arr

    return bart_data, nbart_data


def get_summary(name):
    bart_data = defaultdict(list)
    nbart_data = defaultdict(list)

    for run in range(1, 11):
        run_bart_data, run_nbart_data = get_logs(name, run)

        for k, v in run_bart_data.items():
            bart_data[k].append(v)
        for k, v in run_nbart_data.items():
            nbart_data[k].append(v)

    for k in bart_data.keys():
        bart_data[k] = np.array(bart_data[k])

    for k in nbart_data.keys():
        nbart_data[k] = np.c_[nbart_data[k]].T

    return bart_data, nbart_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    bart_data, nbart_data = get_summary(args.name)

    # plot train
    train_mean = np.mean(bart_data['train_rmse'])

    n, m = nbart_data['train_rmse'].shape
    train_rmse = np.zeros((n * m, 2))
    for i in range(m):
        train_rmse[i * n: (i + 1) * n, 0] = np.arange(n)
        train_rmse[i * n: (i + 1) * n, 1] = nbart_data['test_rmse'][:, i]
    train_rmse = pd.DataFrame(train_rmse, columns=['Training epoch', 'RMSE'])

    sns.lineplot(data=train_rmse, x='Training epoch', y='RMSE', color='green', label='NBART')
    plt.hlines(train_mean, 0, 100, colors='red', label='BART Baseline')
    plt.title("Evolution of training RMSE on {}".format(args.name))
    plt.legend()
    plt.savefig('plots/{}_train.png'.format(args.name))
    plt.clf()

    # plot val
    val_mean = np.mean(bart_data['val_rmse'])

    n, m = nbart_data['val_rmse'].shape
    val_rmse = np.zeros((n * m, 2))
    for i in range(m):
        val_rmse[i * n: (i + 1) * n, 0] = np.arange(n)
        val_rmse[i * n: (i + 1) * n, 1] = nbart_data['test_rmse'][:, i]
    val_rmse = pd.DataFrame(train_rmse, columns=['Training epoch', 'RMSE'])

    sns.lineplot(data=val_rmse, x='Training epoch', y='RMSE', color='blue', label='NBART')
    plt.hlines(train_mean, 0, 100, colors='red', label='BART Baseline')
    plt.title("Evolution of validation RMSE on {}".format(args.name))
    plt.legend()
    plt.savefig('plots/{}_val.png'.format(args.name))
    plt.clf()

    # get test mean and std
    bart_test_mean = np.mean(bart_data['test_rmse'])
    bart_test_std = np.std(bart_data['test_rmse'])

    nbart_test_mean = np.mean(nbart_data['test_rmse'][-1])
    nbart_test_std = np.std(nbart_data['test_rmse'][-1])

    print("BART RMSE: {} ({})".format(bart_test_mean, bart_test_std))
    print("NBART RMSE: {} ({})".format(nbart_test_mean, nbart_test_std))
