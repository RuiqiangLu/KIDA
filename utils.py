import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pathlib
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import mean_squared_error, r2_score
import random
from itertools import compress
from collections import defaultdict

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except:
    MurckoScaffold = None
    print('Please install rdkit for dataset processing')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        # important to add reduction='none' to keep per-batch-item loss
        # alpha=0.25, gamma=2 from https://www.cnblogs.com/qi-yuan-008/p/11992156.html
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class MultiTargetCrossEntropy(torch.nn.Module):
    def __init__(self, C_dim=2):
        super(MultiTargetCrossEntropy, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=C_dim)
        self.nll_loss = torch.nn.NLLLoss()

    def forward(self, input, target):
        # input = input.view(target.shape[0], self.C, target.shape[1])
        '''
            input: shoud be `(N, T, C)` where `C = number of classes` and `T = number of targets`
            target: should be `(N, T)` where `T = number of targets`
        '''
        assert input.shape[0] == target.shape[0]
        assert input.shape[1] == target.shape[1]
        out = self.log_softmax(input)
        out = self.nll_loss(out, target)
        return out


class ChamferLoss(nn.Module):
    def __init__(self, input_channels, reduction="mean"):
        # TODO: repsect reduction rule
        super(ChamferLoss, self).__init__()
        self.input_channels = input_channels

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_points = min(x.shape[1], y.shape[1])
        x = x[:, :num_points, : self.input_channels]
        y = y[:, :num_points, : self.input_channels]
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        indices = torch.arange(0, num_points)
        rx = xx[:, indices, indices].unsqueeze(1).expand([batch_size, num_points, num_points])
        ry = yy[:, indices, indices].unsqueeze(1).expand([batch_size, num_points, num_points])
        pdist = rx.transpose(2, 1) + ry - 2 * zz
        loss = torch.mean(pdist.min(1)[0]) + torch.mean(pdist.min(2)[0])
        return torch.mul(loss, 0.005)


def mdn_loss_fn(sigma, mu, dist):
    normal = torch.distributions.Normal(mu, sigma)
    loglik = normal.log_prob(dist)
    return -loglik


def get_loss(loss_str):
    d = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'huber': torch.nn.SmoothL1Loss(),
        'smae': torch.nn.SmoothL1Loss(),
        'bce': torch.nn.BCELoss(),
        'bcen': torch.nn.BCELoss(reduction="none"),
        'bcel': torch.nn.BCEWithLogitsLoss(),
        'bceln': torch.nn.BCEWithLogitsLoss(reduction="none"),
        'mtce': MultiTargetCrossEntropy(),
        'kl': torch.nn.KLDivLoss(),
        'hinge': torch.nn.HingeEmbeddingLoss(),
        'nll': torch.nn.NLLLoss(),
        'ce': torch.nn.CrossEntropyLoss(),
        'focal': FocalLoss(alpha=0.25),
        'chamfer_huber': {'chamfer': ChamferLoss(input_channels=7),
                          'huber': torch.nn.SmoothL1Loss()},
        'chamfer_ce': {'chamfer': ChamferLoss(input_channels=7),
                       'ce': torch.nn.CrossEntropyLoss()}
    }
    if loss_str not in d.keys():
        raise ValueError('loss not found')
    return d[loss_str]


def cal_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def regression_metrics(y_true, y_pred):
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        pearson_value = np.corrcoef(y_true, y_pred)[0, 1]
        mae = np.mean(np.abs(y_true - y_pred))
        sd = np.std(y_pred)
        d = {'mse': mse, 'rmse': rmse, 'pearson': pearson_value, 'mae': mae, 'sd': sd}
    except ValueError as e:
        print(e)
        d = {'mse': -1, 'rmse': -1, 'pearson': -1, 'mae': -1, 'sd': -1}
    return d


def bedroc_score(y_true, y_score, decreasing=True, alpha=80.5):
    # https://github.com/lewisacidic/scikit-chem/blob/master/skchem/metrics.py
    big_n = len(y_true)
    n = sum(y_true == 1)
    if decreasing:
        order = np.argsort(-y_score)
    else:
        order = np.argsort(y_score)

    m_rank = (y_true[order] == 1).nonzero()[0] + 1
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


def logAUC(y_label, y_pred, min_fp=0.001, adjusted=True):
    #   https://github.com/microsoft/IGT-Intermolecular-Graph-Transformer/blob/main/code/utils.py
    fp, tp, thresholds = roc_curve(y_label, y_pred)
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if lam_index != 0:
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.145  # random curve logAUC
    return logauc


def enrichment_factor_single(y_true, y_score, threshold=0.005):
    # https://github.com/gitter-lab/pria_lifechem/blob/1fd892505a/pria_lifechem/evaluation.py
    labels_arr, scores_arr, percentile = y_true, y_score, threshold
    non_missing_indices = np.argwhere(labels_arr != -1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]
    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    indices = np.argsort(scores_arr, axis=0)[::-1][:sample_size]  # get the index positions for these in library
    n_actives = np.nansum(labels_arr)  # count number of positive labels in library
    n_experimental = np.nansum(labels_arr[indices])  # count number of positive labels in subset

    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile  # calc EF at percentile
    else:
        raise Exception('n actives == 0')
    # return n_actives, ef, ef_max
    return ef


def screening_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    y_true, y_score = y_true.reshape(-1), y_score.reshape(-1)
    auc = roc_auc_score(y_true, y_score)
    # if y_pred is None: y_pred = (y_score > threshod).astype(int)
    # acc = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    logauc = logAUC(y_true, y_score, adjusted=False)
    auprc = average_precision_score(y_true, y_score)
    bedroc = bedroc_score(y_true, y_score, alpha=80.5)
    ef_001 = enrichment_factor_single(y_true, y_score, 0.001)
    ef_005 = enrichment_factor_single(y_true, y_score, 0.005)
    ef_01 = enrichment_factor_single(y_true, y_score, 0.01)
    # ef_02 = enrichment_factor_single(y_true, y_score, 0.02)
    ef_05 = enrichment_factor_single(y_true, y_score, 0.05)
    d = {'auc': auc, 'auprc': auprc, 'bedroc': bedroc, 'logauc': logauc,
         'ef01': ef_001, 'ef05': ef_005, 'ef1': ef_01, 'ef5': ef_05, }
    return d


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the dataset.y tensor. Will filter out
    examples with null value in specified task column of the dataset.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in dataset.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[:, task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset


def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle  # , angle2


def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm(np.cross(vector1, vector2))
    return trianglearea


def area_triangle_vertex(vertex1, vertex2, vertex3):
    trianglearea = 0.5 * np.linalg.norm(np.cross(vertex2 - vertex1, vertex3 - vertex1))
    return trianglearea


def cal_angle_area(vector1, vector2):
    return angle(vector1, vector2), area_triangle(vector1, vector2)


def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
        # raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
