import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
import math
import os
'''
random.seed(2)
np.random.seed(2)
os.environ['PYTHONHASHSEED'] = str(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''


def to_torch_sparse_tensor(x, device='cpu'):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    data = x.data

    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values,
                                                x.shape).to(device)

    return th_sparse_tensor


def tensor_from_numpy(x, device='cpu'):
    return torch.from_numpy(x).to(device)


def globally_normalize_bipartite_adjacency(adjacencies, symmetric=False):
    """ 传入的是邻接矩阵的集合，要在不同的关系矩阵上分别计算 """

    row_sum = [np.sum(adj, axis=1) for adj in adjacencies]
    col_sum = [np.sum(adj, axis=0) for adj in adjacencies]
    # 将为0的设置为无穷大，避免除0
    for i in range(len(row_sum)):
        row_sum[i][row_sum[i] == 0] = np.inf
        col_sum[i][col_sum[i] == 0] = np.inf
    degree_row_inv = [1./r for r in row_sum]
    degree_row_inv_sqrt = [1./np.sqrt(r) for r in row_sum]
    degree_col_inv_sqrt = [1./np.sqrt(c) for c in col_sum]
    normalized_adj = []
    if symmetric:
        for i, adj in enumerate(adjacencies):
            normalized_adj.append(np.diag(degree_row_inv_sqrt[i]).dot(adj).dot(np.diag(degree_col_inv_sqrt[i])))
    else:
        for i, adj in enumerate(adjacencies):
            normalized_adj.append(np.diag(degree_row_inv[i]).dot(adj))
    return normalized_adj


def get_k_fold_data(k, data):
    data = data.values
    X, y = data[:, :], data[:, -1]
    #sfolder = StratifiedKFold(n_splits = k, shuffle=True,random_state=1)
    sfolder = StratifiedKFold(n_splits=k, shuffle=True)

    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        train_label.append(y[train])
        test_label.append(y[test])

        # print('Train: %s | test: %s' % (X[train], X[test]))
        # print('label:%s|label:%s' % (y[train], y[test]))
        #print(len(test))
        #print(X[train][:-1,:])
    return train_data, test_data


def AUC(label, prob):
    return roc_auc_score(label, prob)


def true_positive(pred, target):
    return ((pred == 1) & (target == 1)).sum().clone().detach().requires_grad_(False)


def true_negative(pred, target):
    return ((pred == 0) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_positive(pred, target):
    return ((pred == 1) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_negative(pred, target):
    return ((pred == 0) & (target == 1)).sum().clone().detach().requires_grad_(False)


def precision(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def sensitivity(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out

def specificity(pred, target):
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tn/(tn+fp)
    out[torch.isnan(out)] = 0

    return out


def MCC(pred,target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tn+fn)*(tp+fn)*(tn+fp))
    out[torch.isnan(out)] = 0

    return out

def accuracy(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = (tp+tn)/(tp+tn+fn+fp)
    out[torch.isnan(out)] = 0

    return out


def FPR(pred, target):
    fp = false_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    out = fp/(fp+tn)
    out[torch.isnan(out)] = 0
    return out


def TPR(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = tp/(tp+fn)
    out[torch.isnan(out)] = 0
    return out


def printN(pred, target):
    TP = true_positive(pred, target)
    TN = true_negative(pred, target)
    FP = false_positive(pred, target)
    FN = false_negative(pred, target)
    print("TN:{},TP:{},FP:{},FN:{}".format(TN, TP, FP, FN))
    return TP,TN,FP,FN


def performance(tp,tn,fp,fn):
    final_tp = 0
    final_tn = 0
    final_fp = 0
    final_fn = 0
    for i in range(len(tp)):
        final_fn += fn[i]
        final_fp += fp[i]
        final_tn += tn[i]
        final_tp += tp[i]
    print("TN:{},TP:{},FP:{},FN:{}".format(final_tn, final_tp, final_fp, final_fn))
    ACC = (final_tp + final_tn) /float (final_tp + final_tn + final_fn + final_fp)
    Sen = final_tp / float(final_tp+ final_fn)
    Spe = final_tn/float(final_tn+final_fp)
    Pre = final_tp / float(final_tp + final_fp)
    MCC = (final_tp*final_tn-final_fp*final_fn)/float(math.sqrt((final_tp+final_fp)*(final_tn+final_fn)*(final_tp+final_fn)*(final_tn+final_fp)))
    FPR = final_fp/float(final_fp+final_tn)
    return ACC,Sen, Spe,Pre,MCC,FPR




