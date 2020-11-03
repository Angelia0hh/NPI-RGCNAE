import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from model2 import StackGCNEncoder, FullyConnected, Decoder
from utils import *

######hyper
pr_num = 449
nc_num = 4636
DEVICE = torch.device('cpu')
LEARNING_RATE = 1e-3
EPOCHS = 100
NODE_INPUT_DIM = 5085
SIDE_FEATURE_DIM = 343
GCN_HIDDEN_DIM = 512
SIDE_HIDDEN_DIM = 64
ENCODE_HIDDEN_DIM = 128
NUM_BASIS = 2
DROPOUT_RATIO = 0.7
WEIGHT_DACAY = 0.005
######hyper


SCORES = torch.tensor([0,1]).to(DEVICE)


def to_torch_sparse_tensor(x, device='cpu'):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    data = x.data

    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse.FloatTensor(indices, values,
                                                x.shape).to(device)

    return th_sparse_tensor


def tensor_from_numpy(x, device='cpu'):
    return torch.from_numpy(x).to(device)


class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim,
                 gcn_hidden_dim, side_hidden_dim,
                 encode_hidden_dim,
                 num_support=2, num_classes=2, num_basis=3):
        super(GraphMatrixCompletion, self).__init__()
        self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support)
        self.dense1 = FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                     use_bias=True)
        self.dense2 = FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                     dropout=DROPOUT_RATIO, activation=lambda x: x)
        self.decoder = Decoder(encode_hidden_dim, num_basis, num_classes,
                               dropout=DROPOUT_RATIO, activation=lambda x: x)

    def forward(self, user_supports, item_supports,
                user_inputs, item_inputs,
                user_side_inputs, item_side_inputs,
                user_edge_idx, item_edge_idx):
        user_gcn, movie_gcn = self.encoder(user_supports, item_supports, user_inputs, item_inputs)
        user_side_feat, movie_side_feat = self.dense1(user_side_inputs, item_side_inputs)

        user_feat = torch.cat((user_gcn, user_side_feat), dim=1)
        movie_feat = torch.cat((movie_gcn, movie_side_feat), dim=1)

        user_embed, movie_embed = self.dense2(user_feat, movie_feat)

        edge_logits = self.decoder(user_embed, movie_embed, user_edge_idx, item_edge_idx)

        return edge_logits


'''
user2movie_adjacencies, movie2user_adjacencies, \
user_side_feature, movie_side_feature, \
user_identity_feature, movie_identity_feature, \
user_indices, movie_indices, labels, train_mask = data.build_graph(
    *data.read_data())

user2movie_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in user2movie_adjacencies]
movie2user_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in movie2user_adjacencies]
user_side_feature = tensor_from_numpy(user_side_feature, DEVICE).float()
movie_side_feature = tensor_from_numpy(movie_side_feature, DEVICE).float()
user_identity_feature = tensor_from_numpy(user_identity_feature, DEVICE).float()
movie_identity_feature = tensor_from_numpy(movie_identity_feature, DEVICE).float()
user_indices = tensor_from_numpy(user_indices, DEVICE).long()
movie_indices = tensor_from_numpy(movie_indices, DEVICE).long()
labels = tensor_from_numpy(labels, DEVICE)
train_mask = tensor_from_numpy(train_mask, DEVICE)
'''
def load_dataset(filepath='C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\'):
    NPI_pos_matrix = pd.read_csv(filepath+'NPI_pos.csv', header = None).values
    NPI_neg_matrix = pd.read_csv(filepath+'NPI_neg.csv', header = None).values

    protein_side_feature = pd.read_csv(filepath+'Protein3merfeat.csv').values
    RNA_side_feature = pd.read_csv(filepath+'ncRNA4merfeat.csv').values
    supplement = np.zeros((RNA_side_feature.shape[0],87))
    RNA_side_feature = np.concatenate((RNA_side_feature,supplement),axis=1)

    identity_feature = np.identity(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], dtype=np.float32)
    RNA_identity_feature, protein_identity_feature = identity_feature[:NPI_pos_matrix.shape[0]], identity_feature[NPI_pos_matrix.shape[0]:]
    #emb =
    edgelist = pd.read_csv(filepath+'edgelist.csv',header=None)
    return NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, \
           RNA_identity_feature, protein_side_feature, RNA_side_feature, edgelist

def train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
protein_side_feature, RNA_side_feature, edgelist, model, criterion, optimizer):
    # 随机划分k折交叉验证
    train_data, test_data = get_k_fold_data(10, edgelist)
    for i in range(10):
        print("This is the {}th cross validation ".format(i))
        print("The number of train data is {}".format(len(train_data[i])))
        print("The number of valid data is {}".format(len(test_data[i])))
        print(train_data[i])
        # 训练时计算所有的样本对的得分，但是只有训练集中的通过loss进行优化
        RNA_indices = edgelist.values[:,1]
        protein_indices = edgelist.values[:,2]

        #edgelist一共四列：[index,RNA,protein,label]
        mask = np.ones((nc_num,pr_num))
        mask[test_data[i][:,1],test_data[i][:,2]] = 0 #将验证集中的边置0
        tmp_pos = NPI_pos_matrix*mask
        tmp_neg = NPI_neg_matrix*mask

        #分别建RNA到protein邻接矩阵的列表和protein到RNA的邻接矩阵列表
        RNA2protein_adj= []
        RNA2protein_adj.append(tmp_neg)
        RNA2protein_adj.append(tmp_pos)
        protein2RNA_adj = []
        protein2RNA_adj.append(tmp_neg.T)
        protein2RNA_adj.append(tmp_pos.T)

        #将numpy.array 转为Tensor
        RNA2protein_adj= [to_torch_sparse_tensor(adj, DEVICE) for adj in RNA2protein_adj]
        protein2RNA_adj = [to_torch_sparse_tensor(adj, DEVICE) for adj in protein2RNA_adj]
        RNA_side_feature = tensor_from_numpy(RNA_side_feature, DEVICE).float()
        protein_side_feature = tensor_from_numpy(protein_side_feature, DEVICE).float()
        RNA_identity_feature = tensor_from_numpy(RNA_identity_feature, DEVICE).float()
        protein_identity_feature = tensor_from_numpy(protein_identity_feature, DEVICE).float()
        RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
        protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()
        labels = tensor_from_numpy((train_data[i][:,3]+1)/2, DEVICE).long()
        train_mask = tensor_from_numpy(train_data[i][:,0], DEVICE)

        model_inputs = (RNA2protein_adj,  protein2RNA_adj,
                        RNA_identity_feature, protein_identity_feature,
                        RNA_side_feature, protein_side_feature,
                        RNA_indices, protein_indices, )

        model.train()

        for e in range(EPOCHS):
            logits = model(*model_inputs)
            prob = F.softmax(logits, dim=1)
            print("logits:")
            print(logits)
            print("prob:")
            print(prob)
            pred_y = torch.sum(prob * SCORES, dim=1)
            print("prediction:")
            print(pred_y)
            print("label:")
            print(labels)
            loss = criterion(logits[train_mask], labels)
            rmse = expected_rmse(logits[train_mask], labels)
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d}: Loss: {:.4f}, RMSE: {:.4f}".format(e, loss.item(), rmse.item()))
            if (e + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    logits = model(*model_inputs)
                    test_labels = tensor_from_numpy((test_data[i][:, 3]+1)/2, DEVICE).long()
                    test_mask = tensor_from_numpy(test_data[i][:, 0], DEVICE)
                    loss = criterion(logits[test_mask], test_labels)
                    rmse = expected_rmse(logits[test_mask], test_labels)
                    print("==========================")
                    print('test')
                    prob = F.softmax(logits, dim=1)
                    print("logits:")
                    print(logits)
                    print("prob:")
                    print(prob)
                    pred_y = torch.sum(prob * SCORES, dim=1)
                    print("prediction:")
                    print(pred_y)
                    print("label:")
                    print(labels)
                    print('Test On Epoch {}: loss: {:.4f}, Test rmse: {:.4f}'.format(e, loss.item(), rmse.item()))
                    print("==========================")


def expected_rmse(logits, label):
    #true_y = label + 1  # 原来的评分为1~5，作为label时为0~4
    prob = F.softmax(logits, dim=1)
    pred_y = torch.sum(prob * SCORES, dim=1)

    diff = torch.pow(label - pred_y, 2)

    return torch.sqrt(diff.mean())


if __name__ == "__main__":
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = load_dataset()

    model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                                  SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, num_basis=NUM_BASIS).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)
    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,protein_side_feature, RNA_side_feature, edgelist,model, criterion, optimizer)