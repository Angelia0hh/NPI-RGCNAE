import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Three_layer_network(nn.Module):
    def __init__(self, w_in, hidden1_dim, hidden2_dim):
        super(Three_layer_network, self).__init__()
        self.w_in = w_in
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        #self.hidden3_dim = hidden3_dim
        self.layer1 = nn.Linear(self.w_in, hidden1_dim)
        self.layer2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        #self.layer3 = nn.Linear(self.hidden2_dim, self.hidden3_dim)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight,gain=1)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight,gain=1)
        nn.init.zeros_(self.layer2.bias)
        #nn.init.xavier_uniform_(self.layer3.weight,gain=1)
        #nn.init.zeros_(self.layer3.bias)

    def forward(self, X):
        X = self.layer1(X)
        X = F.relu(X)
        X = self.layer2(X)
        X = F.relu(X)
        #X = self.layer3(X)
        #X = F.relu(X)
        return X

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features # 和X的节点特征的维度一样
        self.out_features = out_features #经过仿射变换的特征维度，也是最后节点的特征维度
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        z = self.gc2(x, adj)
        return z

class GCNAutoEncoder(nn.Module):
    def __init__(self,in_dim,feat_dim,hidden_dim,out_dim,dropout_rate,pr_num,nc_num):
        '''

        :param in_dim: 输入的特征维度
        :param feat_dim: 经过统一降维后的特征维度
        :param hidden_dim: 第一层GCN的输出维度，第二层GCN的输入维度
        :param out_dim:第二层GCN输出维度，也是最终学习的节点维度
        :param dropout_rate:
        :param pr_num:protein个数
        :param nc_num:RNA个数
        '''
        super(GCNAutoEncoder,self).__init__()
        self.linear =  nn.Linear(in_dim,feat_dim)
        self.encoder = GCN(feat_dim,hidden_dim,out_dim,dropout_rate)
        self.Rr = Parameter(torch.FloatTensor(1, out_dim))
        self.pr_num = pr_num
        self.nc_num = nc_num
        nn.init.xavier_uniform_(self.Rr, gain=nn.init.calculate_gain('relu'))

    def forward(self, pr_X, RNA_X, adj, samples):
        pr_X = self.linear(pr_X)
        X = torch.cat((RNA_X,pr_X), dim=0)

        #print("pr_X:")
        #print(pr_X)
        #print("RNA_x:")
        #print(RNA_X)

        embeding = self.encoder(X,adj)
        ncRNA = embeding[samples[:,0],:].clone()
        protein = embeding[samples[:,1]+self.nc_num,:].clone()
        logits = torch.sum(protein*self.Rr*ncRNA,dim=1)
        #print("ncRNA:")
        #print(ncRNA)
        #print("protein:")
        #print(protein)
        #print("relationship weight:")
        #print(self.Rr)
        return protein, ncRNA, logits