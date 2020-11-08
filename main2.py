import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from model2 import *
from utils import *


######hyper
pr_num = 449
nc_num = 4636
DEVICE = torch.device('cpu')
INITIAL_LEARNING_RATE = 0.1
EPOCHS = 100
NODE_INPUT_DIM = 1024
SIDE_FEATURE_DIM = 343
GCN_HIDDEN_DIM = 512
SIDE_HIDDEN_DIM = 64
ENCODE_HIDDEN_DIM = 128
NUM_BASIS = 2
DROPOUT_RATIO = 0.7
WEIGHT_DACAY = 0.005
######hyper
SCORES = torch.tensor([-1,1]).to(DEVICE)

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

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim,
                 gcn_hidden_dim, side_hidden_dim,
                 encode_hidden_dim,
                 num_support=2, num_classes=2, num_basis=3, layers=3):
        super(GraphMatrixCompletion, self).__init__()
        self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support, layers, addloop=True, dropout=0.5)
        self.dense1 = FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                     use_bias=True)
        self.dense2 = FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                     dropout=DROPOUT_RATIO, activation=lambda x: x)
        self.decoder = Decoder(encode_hidden_dim, num_basis, num_classes,
                               dropout=DROPOUT_RATIO, activation=lambda x: x)

        print("layers:" + str(layers))
        print("num_basis:"+str(num_basis))
    def forward(self, RNA_supports, protein_supports,
                RNA_inputs, protein_inputs,
                RNA_side_inputs, protein_side_inputs,
                RNA_edge_idx, protein_edge_idx):

        RNA_gcn, movie_gcn = self.encoder(RNA_supports, protein_supports, RNA_inputs, protein_inputs)
        RNA_side_feat, movie_side_feat = self.dense1(RNA_side_inputs, protein_side_inputs)

        RNA_feat = torch.cat((RNA_gcn, RNA_side_feat), dim=1)
        movie_feat = torch.cat((movie_gcn, movie_side_feat), dim=1)

        RNA_embed, movie_embed = self.dense2(RNA_feat, movie_feat)

        edge_logits = self.decoder(RNA_embed, movie_embed, RNA_edge_idx, protein_edge_idx)

        return edge_logits



def load_dataset(identity_feature, negaive_random_sample, filepath='C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\'):
    NPI_pos_matrix = pd.read_csv(filepath+'NPI_pos.csv', header = None).values
    if (negaive_random_sample==True):
        NPI_neg_matrix = pd.read_csv(filepath + 'NPI_neg_selected_random.csv', header=None).values
    else:
        NPI_neg_matrix = pd.read_csv(filepath+'NPI_neg.csv', header = None).values
    protein_side_feature = pd.read_csv(filepath+'Protein3merfeat.csv').values
    RNA_side_feature = pd.read_csv(filepath+'ncRNA4merfeat.csv').values
    supplement = np.zeros((RNA_side_feature.shape[0],87))
    RNA_side_feature = np.concatenate((RNA_side_feature,supplement),axis=1)


    if (identity_feature == 'one hot'):
        identity_feature = np.identity(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], dtype=np.float32)
        RNA_identity_feature, protein_identity_feature = identity_feature[:NPI_pos_matrix.shape[0]], identity_feature[NPI_pos_matrix.shape[0]:]
    elif (identity_feature == 'node2vec'):
        name = ['index']
        for i in range(1024):
            name.append(i + 1)
        emb = pd.read_csv(filepath+'emb2.emd.txt', header=None, sep=' ', names=name)
        emb.sort_values('index', inplace=True)
        emb = emb[list(range(1, 1025))].values
        RNA_identity_feature, protein_identity_feature = emb[:NPI_pos_matrix.shape[0]],emb[NPI_pos_matrix.shape[0]:]

    edgelist = pd.read_csv(filepath+'edgelist.csv',header=None)
    return NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, \
           RNA_identity_feature, protein_side_feature, RNA_side_feature, edgelist

def train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
protein_side_feature, RNA_side_feature, edgelist, threshold):

    valid_acc = []
    valid_pre= []
    valid_rec = []
    valid_FPR = []
    valid_TPR = []
    valid_loss = []
    valid_RMSE = []
    valid_AUC = []

    # 随机划分k折交叉验证
    train_data, test_data = get_k_fold_data(10, edgelist)
    print(train_data[0])

    # 训练时计算所有的样本对的得分，但是只有训练集中的通过loss进行优化
    RNA_indices = edgelist.values[:, 1]
    protein_indices = edgelist.values[:, 2]
    RNA_side_feature = tensor_from_numpy(RNA_side_feature, DEVICE).float()
    protein_side_feature = tensor_from_numpy(protein_side_feature, DEVICE).float()
    RNA_identity_feature = tensor_from_numpy(RNA_identity_feature, DEVICE).float()
    protein_identity_feature = tensor_from_numpy(protein_identity_feature, DEVICE).float()
    RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
    protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()

    for i in range(10):
        train_pos_num =  len(np.where(train_data[i][:,3]==1)[0])
        train_neg_num = len(np.where(train_data[i][:,3]==-1)[0])
        test_pos_num = len(np.where(test_data[i][:,3]==1)[0])
        test_neg_num = len(np.where(test_data[i][:,3]==-1)[0])
        print("This is the {}th cross validation ".format(i+1))
        print("The number of train data is {},containing positive samples:{} and negative samples:{}"
              .format(len(train_data[i]),train_pos_num,train_neg_num))
        print("The number of valid data is {},containing positive samples:{} and negative samples:{}"
              .format(len(test_data[i]),test_pos_num,test_neg_num))

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
        print("The number of negative samples in the matrix:")
        print(tmp_neg.sum())
        print("The number of positive samples in the matrix:")
        print(tmp_pos.sum())
        RNA2protein_adj = globally_normalize_bipartite_adjacency(RNA2protein_adj,False)
        protein2RNA_adj = globally_normalize_bipartite_adjacency(protein2RNA_adj,False)

        #将numpy.array 转为Tensor
        RNA2protein_adj= [to_torch_sparse_tensor(adj, DEVICE) for adj in RNA2protein_adj]
        protein2RNA_adj = [to_torch_sparse_tensor(adj, DEVICE) for adj in protein2RNA_adj]
        labels = tensor_from_numpy((train_data[i][:,3]), DEVICE).long()
        train_mask = tensor_from_numpy(train_data[i][:,0], DEVICE)

        model_inputs = (RNA2protein_adj,  protein2RNA_adj,
                        RNA_identity_feature, protein_identity_feature,
                        RNA_side_feature, protein_side_feature,
                        RNA_indices, protein_indices, )

        model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                                      SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, num_basis=NUM_BASIS, layers=3).to(DEVICE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DACAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7, last_epoch=-1)
        print("初始化的学习率：", optimizer.defaults['lr'])

        model.train()

        for e in range(EPOCHS):
            logits = model(*model_inputs)
            prob = F.softmax(logits, dim=1).detach()
            pred_y = torch.sum(prob * SCORES, dim=1).detach()
            #y = top_M_and_N(train_pos_num,train_neg_num,pred_y) #所有的样本对对应的结果
            print("The {}th training,the whole number of train dataset is {}:".format(e+1,len(train_mask)))
            print("probability:")
            print(prob)
            print("prediction:")
            print(pred_y[train_mask])
            pred_y[pred_y > threshold] = 1
            pred_y[pred_y <=threshold] = 0
            y = pred_y
            print(pred_y[train_mask])
            print("label:")
            print(labels)
            loss = criterion(logits[train_mask], (labels+1)//2)
            rmse = expected_rmse(logits[train_mask], labels)
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            print("第%d个epoch的学习率：%f" % (e+1, optimizer.param_groups[0]['lr']))
            scheduler.step()

            print("Epoch {:03d}: Loss: {:.4f}, RMSE: {:.4f}".format(e, loss.item(), rmse.item()))
            printN(y[train_mask],(labels+1)//2)
            train_acc = accuracy(y[train_mask],(labels+1)//2)
            train_pre = precision(y[train_mask],(labels+1)//2)
            train_rec = recall(y[train_mask],(labels+1)//2)
            train_FPR = FPR(y[train_mask],(labels+1)//2)
            train_TPR = TPR(y[train_mask],(labels+1)//2)
            print("accuracy:{}, precision:{}, recall:{}, FPR:{}, TPR:{}".
                  format(train_acc,train_pre,train_rec,train_FPR,train_TPR))
            print('\n')

            #验证
            if (e + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    logits = model(*model_inputs)
                    test_labels = tensor_from_numpy(test_data[i][:, 3], DEVICE).long()
                    test_mask = tensor_from_numpy(test_data[i][:, 0], DEVICE)
                    prob = F.softmax(logits, dim=1).detach()
                    pred_y = torch.sum(prob * SCORES, dim=1).detach()
                    #y = top_M_and_N(test_pos_num, test_neg_num, pred_y)
                    loss = criterion(logits[test_mask], (test_labels+1)//2)
                    rmse = expected_rmse(logits[test_mask], test_labels)
                    print("==========================")
                    print('validation')
                    print("probability:")
                    print(prob)
                    print("prediction:")
                    print(pred_y[test_mask])
                    pred_y[pred_y > threshold] = 1
                    pred_y[pred_y <=threshold] = 0
                    y = pred_y
                    print(pred_y[test_mask])
                    print("label:")
                    print(test_labels)
                    print('Test On Epoch {}: loss: {:.4f}, Test rmse: {:.4f}'.format(e, loss.item(), rmse.item()))
                    printN(y[test_mask], (test_labels+1)//2)
                    test_AUC = AUC(test_data[i][:, 3].squeeze(), pred_y[test_mask].detach().numpy().squeeze())
                    test_acc = accuracy(y[test_mask], (test_labels+1)//2)
                    test_pre = precision(y[test_mask], (test_labels+1)//2)
                    test_rec = recall(y[test_mask], (test_labels+1)//2)
                    test_FPR = FPR(y[test_mask], (test_labels+1)//2)
                    test_TPR = TPR(y[test_mask], (test_labels+1)//2)
                    print("accuracy:{}, precision:{}, recall:{}, FPR:{}, TPR:{}, AUC:{}".
                          format(test_acc, test_pre, test_rec, test_FPR, test_TPR, test_AUC))
                    print("==========================")
                    print('\n')
                if (e+1)==100:
                    valid_acc.append(test_acc)
                    valid_pre.append(test_pre)
                    valid_rec.append(test_rec)
                    valid_FPR.append(test_FPR)
                    valid_TPR.append(test_TPR)
                    valid_AUC.append(test_AUC)
                    valid_loss.append(loss)
                    valid_RMSE.append(rmse)

    print("average accuracy:"+str(np.mean(valid_acc)))
    print(valid_acc)
    print("average precision:" + str(np.mean(valid_pre)))
    print(valid_pre)
    print("average recall:" + str(np.mean(valid_rec)))
    print(valid_rec)
    print("average FPR:" + str(np.mean(valid_FPR)))
    print(valid_FPR)
    print("average TPR:" + str(np.mean(valid_TPR)))
    print(valid_TPR)
    print("average AUC:"+str(np.mean(valid_AUC)))
    print(valid_AUC)

def expected_rmse(logits, label):
    prob = F.softmax(logits, dim=1)
    pred_y = torch.sum(prob * SCORES, dim=1)
    diff = torch.pow(label - pred_y, 2)

    return torch.sqrt(diff.mean())


if __name__ == "__main__":
    set_seed(1)
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = load_dataset(identity_feature='node2vec', negaive_random_sample=True)
    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,protein_side_feature, RNA_side_feature, edgelist,
          threshold=0)