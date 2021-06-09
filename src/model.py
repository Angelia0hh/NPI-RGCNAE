import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import torch.nn.init as init
import os
import random

'''
random.seed(2)
np.random.seed(2)
os.environ['PYTHONHASHSEED'] = str(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''


class StackGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, layers, addloop, dropout,
                 use_bias=False, activation=F.relu):
        """对得到的每类评分使用级联的方式进行聚合

        Args:
        ----
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 评分的类别数
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(StackGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.use_bias = use_bias
        self.activation = activation
        assert output_dim % num_support == 0
        self.drop = nn.Dropout(dropout)
        # self.weight = nn.Parameter(torch.Tensor(num_support,
        #                                        input_dim, output_dim // num_support))
        self.addloop = addloop
        self.layers = layers
        if self.addloop:
            self.self_weights = []
            dim1 = input_dim
            dim2 = dim1//2
            for i in range(self.layers):
                if i==0:
                    self.self_weight = nn.Parameter(torch.Tensor(dim1, dim2))
                    self.self_weights.append(self.self_weight)
                else:
                    self.self_weights.append(nn.Parameter(torch.Tensor(dim1, dim2)))
                dim1 = dim2
                dim2 = dim1//2

        self.weights = []
        dim1 = input_dim
        dim2 = dim1//2
        for i in range(self.layers):
            self.weights.append(nn.Parameter(torch.Tensor(num_support, dim1, dim2 // num_support)))
            dim1 = dim2
            dim2 = dim1//2

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_protein = nn.Parameter(torch.Tensor(output_dim, ))
        self.reset_parameters()
        '''
        print("w:")
        for i,w in enumerate(self.weights):
            print(self.weights[i].shape)
        print("selfloop w:")
        for i,w in enumerate(self.self_weights):
            print(self.self_weights[i].shape)
        '''


    def reset_parameters(self):
        for weight in self.weights:
            init.kaiming_uniform_(weight)
        if self.addloop:
            for i in range(self.layers):
                if i==0:
                    init.kaiming_uniform_(self.self_weight)
                else:
                    init.kaiming_uniform_(self.self_weights[i])

        if self.use_bias:
            init.zeros_(self.bias)
            init.zeros_(self.bias_protein)

    def forward(self, RNA_supports, protein_supports, RNA_inputs, protein_inputs):
        """StackGCNEncoder计算逻辑

        Args:
            RNA_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的RNA与protein邻接矩阵
            protein_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的protein与RNA邻接矩阵
            RNA_inputs (torch.Tensor): RNA特征的输入
            protein_inputs (torch.Tensor): protein特征的输入

        Returns:
            [torch.Tensor]: RNA的隐层特征
            [torch.Tensor]: protein的隐层特征
        """

        assert len(RNA_supports) == len(protein_supports) == self.num_support
        # print("initial:")
        H_RNA = RNA_inputs
        H_protein = protein_inputs
        # print(H_RNA.shape)
        # print(H_protein.shape)
        for l in range(self.layers):
            # print(str(l)+" layers")
            RNA_hidden = []
            protein_hidden = []
            for i in range(self.num_support):
                # print(str(i)+" edges")
                # 在每个关系R上分别学习RNA和蛋白质的节点表示，先对同一类型的进行聚合
                tmp_u = torch.matmul(H_RNA, self.weights[l][i])
                tmp_v = torch.matmul(H_protein, self.weights[l][i])
                tmp_RNA_hidden = torch.sparse.mm(RNA_supports[i], tmp_v)
                tmp_protein_hidden = torch.sparse.mm(protein_supports[i], tmp_u)
                RNA_hidden.append(tmp_RNA_hidden)
                protein_hidden.append(tmp_protein_hidden)
                # print("RNA"+str(i)+":"+str(tmp_RNA_hidden.shape))
                # print("protein"+str(i)+":"+str(tmp_protein_hidden.shape))
            # 对不同类型的图学习到的节点特征进行聚合
            RNA_hidden = torch.cat(RNA_hidden, dim=1)
            protein_hidden = torch.cat(protein_hidden, dim=1)
            if self.addloop:
                # 加自连接
                RNA_hidden = RNA_hidden + torch.matmul(H_RNA, self.self_weights[l])
                protein_hidden = protein_hidden + torch.matmul(H_protein, self.self_weights[l])

            RNA_hidden = self.drop(RNA_hidden)
            protein_hidden = self.drop(protein_hidden)
            RNA_outputs = self.activation(RNA_hidden)
            protein_outputs = self.activation(protein_hidden)

            if self.use_bias:
                RNA_outputs += self.bias
                protein_outputs += self.bias_protein

            H_RNA = RNA_outputs
            H_protein = protein_outputs
            #print("H_RNA:"+str(H_RNA.shape))
            #print("H_protein:"+str(H_protein.shape))
            #print()
        return H_RNA, H_protein


class SumGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, layers, dropout, addloop,
                 use_bias=False, activation=F.relu):
        """对得到的每类评分使用求和的方式进行聚合

        Args:
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 2
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(SumGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.addloop = addloop
        if addloop:
            self.W_loops = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weights = []
        for l in range(self.layers):
            self.weights.append(nn.Parameter(torch.Tensor(2, input_dim, output_dim)))
            input_dim = output_dim

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim, ))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight)
        for l in range(len(self.weights)):
            init.kaiming_uniform_(self.weights[l])
        if self.addloop:
            init.kaiming_uniform_(self.W_loops)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, RNA_supports, protein_supports, RNA_inputs, protein_inputs):
        """SumGCNEncoder计算逻辑

        Args:
            RNA_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的RNA与protein邻接矩阵
            protein_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的protein与RNA邻接矩阵
            RNA_inputs (torch.Tensor): RNA特征的输入
            protein_inputs (torch.Tensor): protein特征的输入

        Returns:
            [torch.Tensor]: RNA的隐层特征
            [torch.Tensor]: protein的隐层特征
        """
        assert len(RNA_supports) == len(protein_supports) == self.num_support
        H_RNA = RNA_inputs
        H_protein = protein_inputs
        for l in range(self.layers):
            RNA_hidden = 0
            protein_hidden = 0
            for i in range(self.num_support):
                tmp_u = torch.matmul(H_RNA, self.weights[l][i])
                tmp_v = torch.matmul(H_protein, self.weights[l][i])
                tmp_RNA_hidden = torch.sparse.mm(RNA_supports[i], tmp_v)
                tmp_protein_hidden = torch.sparse.mm(protein_supports[i], tmp_u)
                RNA_hidden += tmp_RNA_hidden
                protein_hidden += tmp_protein_hidden
            if self.addloop:
                RNA_hidden = RNA_hidden + torch.matmul(RNA_inputs, self.W_loops)
                protein_hidden = protein_hidden + torch.matmul(protein_inputs, self.W_loops)

            protein_hidden = self.dropout(protein_hidden)
            RNA_hidden = self.dropout(RNA_hidden)

            RNA_outputs = self.activation(RNA_hidden)
            protein_outputs = self.activation(protein_hidden)

            if self.use_bias:
                RNA_outputs += self.bias
                protein_outputs += self.bias_protein
            H_RNA = RNA_outputs
            H_protein = protein_outputs
        return H_RNA, H_protein


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.,
                 use_bias=False, activation=F.relu,
                 share_weights=False):
        """非线性变换层

        Args:
        ----
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
            share_weights (bool, optional): RNA和protein是否共享变换权值. Defaults to False.

        """
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = activation
        self.share_weights = share_weights
        self.linear_RNA = nn.Linear(input_dim, output_dim, bias=use_bias)
        if self.share_weights:
            self.linear_protein = self.linear_RNA
        else:
            self.linear_protein = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, RNA_inputs, protein_inputs):
        """前向传播

        Args:
            RNA_inputs (torch.Tensor): 输入的RNA特征
            protein_inputs (torch.Tensor): 输入的protein特征

        Returns:
            [torch.Tensor]: 输出的RNA特征
            [torch.Tensor]: 输出的protein特征
        """


        RNA_inputs = self.dropout(RNA_inputs)
        RNA_outputs = self.linear_RNA(RNA_inputs)

        protein_inputs = self.dropout(protein_inputs)
        protein_outputs = self.linear_protein(protein_inputs)

        if self.activation:
            RNA_outputs = self.activation(RNA_outputs)
            protein_outputs = self.activation(protein_outputs)

        return RNA_outputs, protein_outputs


class Decoder(nn.Module):
    def __init__(self, input_dim, num_weights, num_classes, dropout=0., activation=F.relu):
        """解码器

        Args:
        ----
            input_dim (int): 输入的特征维度
            num_weights (int): basis weight number
            num_classes (int): 总共的评分级别数
        """
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_weights = num_weights
        self.num_classes = num_classes
        self.activation = activation

        # self.weight = nn.Parameter(torch.Tensor(num_weights, input_dim, input_dim))
        self.weight_classifier = nn.Parameter(torch.Tensor(num_weights, num_classes))
        self.w_relation = nn.Parameter(torch.Tensor(num_classes, input_dim))
        self.reset_parameters()

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.weight_classifier)
        init.kaiming_uniform_(self.w_relation)

    def forward(self, RNA_inputs, protein_inputs, RNA_indices, protein_indices):
        """计算非归一化的分类输出

        Args:
            RNA_inputs (torch.Tensor): RNA的特征
            protein_inputs (torch.Tensor): protein的特征
            RNA_indices (torch.LongTensor):
                所有已知互作中中RNA的id索引，与对应的protein_indices构成一条边,shape=(num_edges, )
            protein_indices (torch.LongTensor):
                所有已知互作中protein的id索引，与对应的RNA_indices构成一条边,shape=(num_edges, )

        Returns:
            [torch.Tensor]: 未归一化的分类输出，shape=(num_edges, num_classes)
        """
        RNA_inputs = self.dropout(RNA_inputs)
        protein_inputs = self.dropout(protein_inputs)
        RNA_inputs = RNA_inputs[RNA_indices]
        protein_inputs = protein_inputs[protein_indices]
        '''
        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(RNA_inputs, self.weight[i])
            out = torch.sum(tmp * protein_inputs, dim=1, keepdim=True)
            basis_outputs.append(out)
        basis_outputs = torch.cat(basis_outputs, dim=1)
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)

        '''
        basis_outputs = []
        for i in range(self.num_classes):
            basis_output = torch.sum(RNA_inputs * self.w_relation[i, :] * protein_inputs, dim=1, keepdim=True)
            basis_outputs.append(basis_output)
        basis_outputs = torch.cat(basis_outputs, dim=1)
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)

        return outputs


class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim,
                 gcn_hidden_dim, side_hidden_dim,
                 encode_hidden_dim, dropout, use_side_feature, accumulate_strategy,
                 num_support=2, num_classes=2, num_basis=2, layers=3):
        super(GraphMatrixCompletion, self).__init__()
        self.use_side_feature = use_side_feature
        if accumulate_strategy == 'stack':
            self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support, layers, addloop=True,
                                           dropout=dropout)
        elif accumulate_strategy == 'sum':
            self.encoder = SumGCNEncoder(input_dim, gcn_hidden_dim, num_support, layers, addloop=True, dropout=dropout)

        if self.use_side_feature:
            # 处理side_feature
            self.dense1 = FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                         use_bias=True)
            # 结合side_feature和embeddings
            self.dense2 = FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                         dropout=dropout, activation=lambda x: x)
        else:
            self.dense2 = FullyConnected(gcn_hidden_dim, encode_hidden_dim,
                                         dropout=dropout, activation=lambda x: x)
        self.decoder = Decoder(encode_hidden_dim, num_basis, num_classes,
                               dropout=dropout, activation=lambda x: x)

    def forward(self, RNA_supports, protein_supports,
                RNA_inputs, protein_inputs,
                RNA_side_inputs, protein_side_inputs,
                RNA_edge_idx, protein_edge_idx):

        RNA_gcn, protein_gcn = self.encoder(RNA_supports, protein_supports, RNA_inputs, protein_inputs)


        if self.use_side_feature:
            RNA_side_feat, protein_side_feat = self.dense1(RNA_side_inputs, protein_side_inputs)

            RNA_feat = torch.cat((RNA_gcn, RNA_side_feat), dim=1)
            protein_feat = torch.cat((protein_gcn, protein_side_feat), dim=1)
        else:
            RNA_feat = RNA_gcn
            protein_feat = protein_gcn

        RNA_embed, protein_embed = self.dense2(RNA_feat, protein_feat)

        edge_logits = self.decoder(RNA_embed, protein_embed, RNA_edge_idx, protein_edge_idx)

        return edge_logits