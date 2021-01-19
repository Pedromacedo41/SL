from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


class TreeNode(object):
    """Represents a node from a BART tree."""
    def __init__(
        self, n_eta=None, split_attribute_M= 0, split_value=0, y_pred=None,
        y_avg=None, posterior_var=None, posterior_mean=None, left=None,
        right=None
    ):
        self.n_eta = n_eta
        self.left = left
        self.right = right

        # attributes for internal nodes
        self.split_attribute_M = split_attribute_M
        self.split_value = split_value

        # attributes for leaves
        self.y_pred = y_pred
        self.y_avg = y_avg
        self.posterior_var = posterior_var
        self.posterior_mean = posterior_mean


class LinearNoReset(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearNoReset, self).__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        pass

class NBART(nn.Module):
    """Neural Network derived from a single BART tree."""
    def __init__(self, input_dim, inner_decisions, paths, gamma1=100.0, gamma2=1.0):
        super(NBART, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.dic = {}
        self.lin1 = LinearNoReset(input_dim, len(inner_decisions))
        self.set_connections_1_layer(inner_decisions)

        self.lin2 = LinearNoReset(len(inner_decisions), 1 + len(inner_decisions))
        self.avgs = []
        self.set_connections_2_layer(paths)

        self.lin3 = LinearNoReset(1 + len(inner_decisions), 1)
        self.set_connections_3_layer()

    def forward(self, x):
        out = self.lin1(x)
        out = self.lin2(torch.tanh(self.gamma1 * out))
        return self.lin3(torch.tanh(self.gamma2 * out))

    def set_connections_1_layer(self, inner_decisions):
        with torch.no_grad():
            self.lin1.weight.data = torch.zeros(self.lin1.weight.shape)
            self.lin1.bias.data = torch.zeros(self.lin1.bias.shape)
            for i, elem in enumerate(inner_decisions):
                # node k to layer index
                self.dic[elem[0]] = i

                self.lin1.weight.data[i, int(elem[1])] = 1
                self.lin1.bias[i] = elem[2]

    def set_connections_2_layer(self, paths):
        with torch.no_grad():
            self.lin2.weight.data = torch.zeros(self.lin2.weight.shape)
            self.lin2.bias.data = torch.zeros(self.lin2.bias.shape)
            for i, elem in enumerate(paths):
                _, avg, tuples = elem
                self.avgs.append(avg)
                self.lin2.bias[i] = (-len(tuples) + 1/2)
                for a in tuples:
                    self.lin2.weight.data[i, self.dic[a[0]]] = a[1]

    def set_connections_3_layer(self):
        with torch.no_grad():
            self.lin3.weight.data = (0.5 * torch.as_tensor(self.avgs)).reshape(1, -1)
            self.lin3.bias.data = self.lin3.weight.data.sum()


def inner_decisions(root, leafinfo, k):
    decisions_pairs= []
    all_paths = []
    leafinfos = leafinfo.copy()

    if root.left is None and root.right is None:
        return decisions_pairs, [(k, root.posterior_mean, leafinfos)], k

    decisions_pairs.append((k, root.split_attribute_M, root.split_value))

    leafinfos.append((k, -1))
    dec, paths, K =  inner_decisions(root.left, leafinfos, k + 1)
    decisions_pairs.extend(dec)
    all_paths.extend(paths)

    leafinfos.pop()
    leafinfos.append((k, +1))
    dec, paths , K =  inner_decisions(root.right, leafinfos, K+1)
    decisions_pairs.extend(dec)
    all_paths.extend(paths)

    return decisions_pairs, all_paths, K


def load_trees(filename):
    trees = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            vals = iter(line.split())

            def helper():
                val = next(vals)
                if val == '#':
                    return None

                elems = val.split("@")
                node = TreeNode(n_eta = float(elems[0]), split_attribute_M = float(elems[1]),
                            split_value = float(elems[2]), y_pred=float(elems[3]),
                             y_avg= float(elems[4]),posterior_var= float(elems[5]), posterior_mean=float(elems[6]))

                node.left = helper()
                node.right = helper()
                return node

            trees.append(helper())
    return trees


def load_bart(model_name):
    filenames = glob("models/{}/*.model".format(model_name))
    bart = []
    for filename in filenames:
        bart.append(load_trees(filename))
    return bart


def trees_to_nbart(trees, input_dim):
    nns = []
    for tree in trees:
        decisions, paths, _ =  inner_decisions(tree, [], 0)
        nns.append(NBART(input_dim, decisions, paths))
    return nns


def bart_to_nbart(bart, input_dim):
    nbart = []
    for trees in bart:
        nbart.append(trees_to_nbart(trees, input_dim))
    return nbart


if __name__ == "__main__":
    left = TreeNode(n_eta=1, left = TreeNode(n_eta=2) , right =TreeNode(n_eta=3, left= TreeNode(n_eta=4), right= TreeNode(n_eta=5)))
    right = TreeNode(n_eta=6, left = TreeNode(n_eta=7) , right =TreeNode(n_eta=8, left= TreeNode(n_eta=9), right= TreeNode(n_eta=10)))
    tree = TreeNode(n_eta=0, left=left, right=right)
    print(tree)

    # test
    decisions, paths, K = inner_decisions(tree, [], k = 0)
    print(decisions)
    print(paths)