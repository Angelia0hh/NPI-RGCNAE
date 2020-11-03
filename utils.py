import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random

def sparse_normalization(adj):
    '''计算对称的拉普拉斯矩阵'''
    adj += sp.eye(adj.shape[0]) #增加自连接
    degree = np.array(adj.sum(1)) #度矩阵
    d_hat = sp.diags(np.power(degree, -0.5).flatten()) #建对角阵
    return d_hat.dot(adj).dot(d_hat).tocoo()

def normalization(adj):
    A = adj + np.eye(adj.shape[0])
    degree = A.sum(axis=1)
    D = np.diag(np.power(degree,-0.5))
    return np.dot(np.dot(D,A),D)

#def cal_Gassuian_Interaction_Profile():

def load_NPInter(filepath):
    print('Loading dataset...')
    NPInter = pd.read_table(filepath + 'raw_data\\NPInter10412_dataset.txt')
    protein = NPInter['UNIPROT-ID'].unique().tolist()#蛋白质列表
    ncRNA = NPInter['NONCODE-ID'].unique().tolist()#RNA列表
    pr_num = len(protein)
    nc_num = len(ncRNA)
    NPI = np.zeros((nc_num, pr_num))
    positive_index= [] #相互作用对的下标[ncRNA_index,protein_index,label]
    for ind, row in NPInter.iterrows():
        i = ncRNA.index(row['NONCODE-ID'])
        j = protein.index(row['UNIPROT-ID'])
        NPI[i, j] = 1
        positive_index.append([i,j,1])

    name = ['index']
    for i in range(256):
        name.append(i + 1)
    emb = pd.read_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\emb1.emd', header=None,sep=' ', names=name)
    emb.sort_values('index', inplace=True)
    emb = torch.FloatTensor(emb[list(range(1,257))].values)
    pr3mer = pd.read_csv(filepath+"generated_data\\Protein3merfeat.csv")
    RNA4mer = pd.read_csv(filepath+"generated_data\\ncRNA4merfeat.csv")
    pr3mer = torch.FloatTensor(pr3mer.values)
    RNA4mer = torch.FloatTensor(RNA4mer.values)

    print("---------The dataset Information---------")
    print("ncRNA: "+str(nc_num)+" protein:"+str(pr_num)+" ncRNA_protein interaction:"+str(len(positive_index)))
    print("protein feature shape:"+str(pr3mer.shape))
    print("ncRNA feature shape:" + str(RNA4mer.shape))
    print("The number of observed samples:"+str(len(positive_index)))

    return NPI, emb, pr3mer, RNA4mer, protein, ncRNA, positive_index

'''
def get_k_fold_data(k, edgelist):


    edgelist = edgelist.reindex(np.random.permutation(edgelist.index))
    #random.shuffle(positive)
    #data = pd.DataFrame(positive)
    #data = data.values
    #data = pd.DataFrame(data)
    edgelist.to_csv("C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\shuffle.csv")
    edgelist = edgelist.values
    X,y = edgelist[:,:], edgelist[:,-1]
    sfolder = StratifiedKFold(n_splits = k, shuffle=True, random_state=1)
    train_data = []
    test_data = []
    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        #print('Train: %s | test: %s' % (X[train], X[test]))
        #print('label:%s|label:%s' % (y[train], y[test]))
    return train_data, test_data
'''
def get_k_fold_data(k, positive):
    '''
    postivie/negative(numpy.ndarray)
    n_ratio:the ratio of negative samples
    '''
    positive = pd.DataFrame(positive)
    data = positive.values
    X,y = data[:,:], data[:,-1]
    sfolder = StratifiedKFold(n_splits = k,shuffle=True,random_state=1)

    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        train_label.append(y[train])
        test_label.append(y[test])
        #print('Train: %s | test: %s' % (X[train], X[test]))
        #print('label:%s|label:%s' % (y[train], y[test]))
    return train_data, test_data




def true_positive(pred, target):
    return torch.tensor(((pred == 1) & (target == 1)).sum())

def true_negative(pred, target):
    return torch.tensor(((pred == 0) & (target == 0)).sum())

def false_positive(pred, target):
    return torch.tensor(((pred == 1) & (target == 0)).sum())


def false_negative(pred, target):
    return torch.tensor(((pred == 0) & (target == 1)).sum())


def precision(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def recall(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out


def accuracy(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = (tp+tn)/(tp+tn+fn+fp)
    out[torch.isnan(out)]=0

    return out

def FPR(pred,target):
    fp = false_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    out = fp/(fp+tn)
    out[torch.isnan(out)]=0
    return out


def TPR(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = tp/(tp+fn)
    out[torch.isnan(out)]=0
    return out

def printN(pred, target):
    TP = true_positive(pred, target)
    TN = true_negative(pred, target)
    FP = false_positive(pred, target)
    FN = false_negative(pred, target)
    print("TN:{},TP:{},FP:{},FN:{}".format(TN, TP, FP, FN))

#def AUC(pred, target):

#def AUPR(pred, target):


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def top_K(k,pred):
    y,index = pred.sort(descending=True)
    y[:k] = 1
    y[k:] = 0
    return y,index

