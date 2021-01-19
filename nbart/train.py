from collections import defaultdict
from math import sqrt
import pickle
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from nbart import load_trees, trees_to_nbart


def load_data(name):
    # Load CSV to tensor
    X = pd.read_csv('data/X_{}.csv'.format(name), header=None)
    X = torch.Tensor(X.to_numpy())
    y = pd.read_csv('data/y_{}.csv'.format(name), header=None)
    y = torch.Tensor(y.to_numpy())

    # 50/25/25 split into train/validation/test
    n = len(X)
    train_end, val_end = int(0.5 * n), int(0.75 * n)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    # Create and return datasets
    train = TensorDataset(X_train, y_train)
    val = TensorDataset(X_val, y_val)
    test = TensorDataset(X_test, y_test)
    return train, val, test


def predict(nns, x):
    out = nns[0](x)
    for m in nns[1:]:
        out += m(x)
    return out


def compute_rmse(nbart, loader):
    with torch.no_grad():
        rmse = None
        for x, y in loader:
            y_hat = predict(nbart, x)
            rmse = torch.sqrt(torch.mean((y_hat - y) ** 2)).item()
            break
        return rmse


def log(metrics, scalar_dict, step):
    for k, v in scalar_dict.items():
        metrics[k].append((step, v))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('run', type=str)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    args = parser.parse_args()

    # Prepare directories
    model_name = "nbart_{}{}".format(args.name, args.run)
    os.makedirs('logs/', exist_ok=True)
    os.makedirs('models/', exist_ok=True)

    if os.path.isdir('logs/{}/'.format(model_name)):
        shutil.rmtree('logs/{}/'.format(model_name)) # remove previous logs

    # Load data
    train, val, test = load_data(args.name)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)

    # Set up logs
    metrics = defaultdict(list)
    writer = SummaryWriter('logs/{}'.format(model_name))

    # Get model, optimizer and loss function
    input_dim = train[0][0].shape[0]
    trees = load_trees('models/bart_{}{}.model'.format(args.name, args.run))
    nbart = trees_to_nbart(trees, input_dim)

    params = []
    for m in nbart:
        params.extend(list(m.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    optimizer.zero_grad()

    loss_fn = nn.MSELoss()

    val_rmse = compute_rmse(nbart, val_loader)
    test_rmse = compute_rmse(nbart, test_loader)
    print("start:\n"
          "  val rmse: {}\n"
          "  test rmse: {}\n".format(
            val_rmse, test_rmse))

    for epoch in tqdm.trange(1, args.epochs + 1):
        train_losses = []
        for x, y in train_loader:
            y_hat = predict(nbart, x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().item())

        train_rmse = sqrt(np.mean(np.array(train_losses)))
        val_rmse = compute_rmse(nbart, val_loader)
        test_rmse = compute_rmse(nbart, test_loader)

        writer.add_scalars('rmse', {
            'train': train_rmse,
            'val': val_rmse,
            'test': test_rmse,
        }, epoch)
        metrics = log(metrics, {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
        }, epoch)
        if epoch % 5 == 0:
            print("epoch {}\n"
                  "  train rmse: {}\n"
                  "  val rmse: {}\n"
                  "  test rmse: {}\n".format(
                      epoch, train_rmse, val_rmse, test_rmse))

    with open('logs/{}/metrics.pkl'.format(model_name), 'wb') as f:
        pickle.dump(metrics, f)
    writer.close()