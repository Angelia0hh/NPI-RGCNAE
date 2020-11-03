from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *

# Training settings
'''
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed) 
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''
#parameters setting
filepath = 'C:\\Users\\yuhan\\Desktop\\GNNAE\\'
learning_rate = 0.0001
weight_decay = 0.01
k = 10
epochs = 100
criterion = nn.BCEWithLogitsLoss()
# Load data
NPI, emb, pr3mer, RNA4mer, protein, ncRNA, positive_index = load_NPInter(filepath)
train_data_list, valid_data_list = get_k_fold_data(k, positive_index)
train_data = np.array(train_data_list)
valid_data = np.array(valid_data_list)
pr_num = len(protein)
nc_num = len(ncRNA)

# Model and optimizer
model = GCNAutoEncoder(
     in_dim = emb.shape[1],
     feat_dim = 256,
     hidden_dim = 128,
     out_dim = 64,
     dropout_rate = 0.5,
     pr_num = len(protein),
     nc_num = len(ncRNA))
optimizer = optim.Adam(model.parameters(),
                       lr= learning_rate, weight_decay= weight_decay)
t_total = time.time()
print("...training model...")
for i in range(k):
    print("\n")
    print("**************************")
    print(str(i+1)+"th training: the number of validation is:"+str(len(valid_data[i])))
    #print("train_data:")
    #print(train_data[i])
    #print("valid_data")
    #print(valid_data[i])
    mask = np.ones((nc_num,pr_num)) #实际观测到的邻接矩阵的mask
    mask[valid_data[i][:,0],valid_data[i][:,1]] = 0 #将验证集中的样本置0
    A = np.zeros((pr_num + nc_num, pr_num + nc_num))
    tmp = NPI*mask
    A[:nc_num, nc_num:] = tmp
    A[nc_num:, :nc_num] = tmp.T
    adj = torch.from_numpy(normalization(A)).float()

    X = np.zeros((pr_num+nc_num,pr_num+nc_num))
    tmp1 = tmp.sum(axis=1).reshape(-1,1)
    tmp1[tmp1==0]=1
    tmp2 = tmp.T.sum(axis=1).reshape(-1,1)
    tmp2[tmp2 == 0] = 1
    X[:nc_num,nc_num:] = tmp/tmp1
    X[nc_num:,:nc_num] = tmp.T/tmp2
    X[X== float('inf')] = 0
    X = torch.from_numpy(X).float()



    #index = A.nonzero()
    #adj = sp.coo_matrix((np.ones(len(index[0])), (index[0], index[1])),
    #                     shape=(A.shape[0],A.shape[1]),
    #                     dtype=np.float32)
    #adj = sparse_normalization(adj)
    #adj = sparse_mx_to_torch_sparse_tensor(adj)

    index = tmp.nonzero() #所有非零元素的下标
    label = np.ones(len(index[0])) #构造标签，所有元素为1的向量，长度和观察到的样本数目一样
    #label = tmp.flatten()
    train_sample = []
    for p in range(nc_num):
        for q in range(pr_num):
            train_sample.append([p,q])
    train_sample = np.array(train_sample)

    unobeserved_index = np.where(NPI==0) #所有实际未观测到的样本
    unobeserved = np.zeros((len(unobeserved_index[0]),3))
    print("unobsered is:"+str(len(unobeserved_index[0])))
    unobeserved[:,0] = unobeserved_index[0]
    unobeserved[:,1] = unobeserved_index[1]
    unobeserved[:,2] = 0
    unobeserved = np.concatenate((unobeserved,valid_data[i]),axis=0) #本次循环中未观测到的样本为实际未观测到的和验证集中样本之和

    test_train_data = np.concatenate((unobeserved[:9370,:],train_data[i]),axis=0)



    for j in range(epochs):
        print(str(j+1)+"th epoch")
        model.train()
        optimizer.zero_grad()
        #protein, ncRNA, logits = model.forward(pr_X = pr3mer, RNA_X = RNA4mer, adj = adj, samples = train_data[i])
        logits = model.forward(emb,pr3mer,RNA4mer,adj=adj, samples = test_train_data)
        train_loss = criterion(logits,torch.from_numpy(test_train_data[:,2]).reshape(-1,1))
        pred = torch.sigmoid(logits).detach()
        train_loss.backward()
        optimizer.step()
        print("label:")
        print(test_train_data[:,2])
        #print(label)
        print("train prediction:")
        print(pred)

        pred[pred>=0.7] = 1
        pred[pred<0.7] =0
        #pred,index = top_K(9370,pred)
        #train_acc = accuracy(pred, label)
        #train_pre = precision(pred, label)
        #train_rec = recall(pred, label)
        #train_fpr = FPR(pred, label)
        #train_tpr = TPR(pred, label)
        #printN(pred, label)
        #print('Train - Loss: {},acc:{},precision:{},recall:{},FPR:{},TPR:{}'.format(train_loss, train_acc, train_pre,
        #                                                                                  train_rec,train_fpr,train_tpr))

    '''
        #验证
        if (j+1)%5==0:
        model.eval()
        with torch.no_grad():
            #valid_protein, valid_ncRNA, valid_logits = model.forward(pr3mer,RNA4mer, adj, samples= unobeserved[:,:-1])
            valid_protein, valid_ncRNA, valid_logits = model.forward(emb,pr3mer,RNA4mer,adj=adj, samples= unobeserved[:,:-1])
            valid_label = torch.from_numpy(unobeserved[:, 2])
            valid_loss = criterion(valid_logits,valid_label)
            valid_pred = torch.sigmoid(valid_logits).detach()

            print("**************************")
            print("label:")
            print(valid_label.reshape(2,-1))
            print("valid prediction:")
            print(valid_pred.reshape(2,-1))
            valid_pred, index = top_K(1042, valid_pred)
            valid_label = valid_label[index]
            #valid_pred[valid_pred >= 0.7] = 1
            #valid_pred[valid_pred < 0.7] = 0
            valid_acc = accuracy(valid_pred, valid_label)
            valid_pre = precision(valid_pred, valid_label)
            valid_rec = recall(valid_pred, valid_label)
            valid_fpr = FPR(valid_pred, valid_label)
            valid_tpr = TPR(valid_pred, valid_label)
            printN(valid_pred, valid_label)
            print('Train - Loss: {},acc:{},precision:{},recall:{},FPR:{},TPR:{}'.format(valid_loss, valid_acc,
                                                                                    valid_pre,train_rec, valid_fpr,valid_tpr))
            print("**************************")
    '''



print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

'''
def train(epoch):
   

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
'''
