import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from model import *
from utils import *
import os
import sys
# import matplotlib.pyplot as plt
import time

seed = 1
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cpu')
SCORES = torch.tensor([-1, 1]).to(DEVICE)

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def split_data(dataset):
    data = pd.read_csv('data' + os.sep + 'generated_data' + os.sep + dataset + os.sep + 'edgelist_sort.csv',
                       header=None, names=['index', 'rna', 'protein', 'label'])
    pos = data[data['label']==1]
    neg = data[data['label']==-1]
    train_pos = pos.sample(frac=0.8, random_state=1,replace=False)
    test_pos = pos[~(pos['index'].isin(train_pos['index'].tolist()))]
    train_neg = neg.sample(frac=0.8,random_state=1,replace=False)
    test_neg = neg[~(neg['index'].isin(train_neg['index'].tolist()))]
    '''
    print(len(train_pos))
    print(len(test_pos))
    print(len(train_neg))
    print(len(test_neg))
    print(pd.merge(train_pos, test_pos, on=['index']))
    print(pd.merge(train_neg, test_neg, on=['index']))
    '''
    pr_list = []
    rna_list = []
    with open('data'+os.sep+'generated_data'+os.sep+dataset+os.sep+'protein.txt') as f:
        for pr in f:
            pr_list.append(pr.rstrip())

    with open('data'+os.sep+'generated_data'+os.sep+dataset+os.sep+'ncRNA.txt') as f:
        for rna in f:
            rna_list.append(rna.rstrip())

    edgelist_train = [train_pos, train_neg]
    edgelist_train = pd.concat(edgelist_train, axis=0)
    edgelist_train = edgelist_train.take(np.random.permutation(len(edgelist_train)))  # 打乱正负样本的顺序
    edgelist_train.to_csv('data'+os.sep+'generated_data'+os.sep+'independent'+os.sep+dataset+os.sep+'train'+os.sep+'edgelist_sort.csv', index=False,header=None)

    edgelist_test = [test_pos, test_neg]
    edgelist_test = pd.concat(edgelist_test, axis=0)
    edgelist_test = edgelist_test.take(np.random.permutation(len(edgelist_test)))  # 打乱正负样本的顺序
    edgelist_test.to_csv(
        'data' + os.sep + 'generated_data' + os.sep + 'independent' + os.sep+dataset+os.sep+'test'+os.sep + 'edgelist_sort.csv',index=False,
        header=None)

    NPI_pos = np.zeros((len(rna_list), len(pr_list)))
    NPI_pos[train_pos.values[:, 1], train_pos.values[:, 2]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv( 'data' + os.sep + 'generated_data' + os.sep + 'independent' +os.sep+dataset+os.sep+'train'+ os.sep + 'NPI_pos.csv', index=False, header=None)
    NPI_pos.to_csv('data' + os.sep + 'generated_data' + os.sep + 'independent'+ os.sep+dataset+os.sep+'test' +os.sep+ 'NPI_pos.csv', index=False, header=None)
    print(NPI_pos.values.sum())
    NPI_neg = np.zeros((len(rna_list), len(pr_list)))
    NPI_neg[train_neg.values[:, 1], train_neg.values[:, 2]] = 1
    NPI_neg = pd.DataFrame(NPI_neg)
    NPI_neg.to_csv('data' + os.sep + 'generated_data' + os.sep + 'independent'+os.sep+dataset+os.sep+'train' + os.sep + 'NPI_neg.csv', index=False, header=None)
    NPI_neg.to_csv('data' + os.sep + 'generated_data' + os.sep + 'independent' + os.sep+dataset+os.sep+'test'+ os.sep+ 'NPI_neg.csv', index=False, header=None)
    print(NPI_neg.values.sum())


def load_dataset(dataset, filepath, identity_feature, negative_random_sample, identity_feature_dim=1024):
    filepath = os.path.join(filepath, dataset)
    NPI_pos_matrix = pd.read_csv(os.path.join(filepath, os.path.join('train','NPI_pos.csv')), header=None).values

    name = ['index']
    for i in range(1024):
        name.append(i + 1)

    NPI_neg_matrix = pd.read_csv(os.path.join(filepath, os.path.join('train',"NPI_neg.csv")),
                                 header=None).values
    train_edgelist = pd.read_csv(os.path.join(filepath, os.path.join('train','edgelist_' + negative_random_sample + '.csv')),
                                 names=['index','RNA','protein','label'],header=None)
    test_edgelist = pd.read_csv(os.path.join(filepath, os.path.join('test','edgelist_' + negative_random_sample + '.csv')),
                                names=['index','RNA','protein','label'],header=None)

    #test_edgelist = test_edgelist[test_edgelist['label']==1]
    #train_edgelist = train_edgelist[train_edgelist['label'] == 1]

    if (identity_feature == 'one hot'):
        identity_feature = np.identity(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], dtype=np.float32)
        RNA_identity_feature, protein_identity_feature = identity_feature[
                                                         :NPI_pos_matrix.shape[0]], identity_feature[
                                                                                    NPI_pos_matrix.shape[0]:]
    # elif (identity_feature == 'node2vec'):
    # emb.sort_values('index', inplace=True)  # 按index排序
    # emb = emb[list(range(1, 1025))].values
    # RNA_identity_feature, protein_identity_feature = emb[:NPI_pos_matrix.shape[0]], emb[NPI_pos_matrix.shape[0]:]
    elif (identity_feature == 'random'):
        feature = np.random.randn(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], identity_feature_dim)
        RNA_identity_feature, protein_identity_feature = feature[:NPI_pos_matrix.shape[0]], feature[
                                                                                            NPI_pos_matrix.shape[0]:]

    return NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, \
           RNA_identity_feature, train_edgelist,test_edgelist


def save_load_model(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,train_edgelist,test_edgelist,
                    NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM, SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM,
                    threshold, probsavepath, metricssavepath, embedsavepath, use_side_feature, accumulate_strategy,
                    EPOCHS, DROPOUT_RATIO, INITIAL_LEARNING_RATE, layers, WEIGHT_DACAY, step_size, gamma):
    '''

    :param NPI_pos_matrix: binary matrix of positive samples，shape:[ncRNA_num,protein_num]
    :param NPI_neg_matrix: binary matrix of negative samples， shape:[ncRNA_num, protein_num]
    :param protein_identity_feature: identity feature matirx of proteins, node2vec or one hot, shape:[protein_num, identity_feature_dim]
    :param RNA_identity_feature: identity feature matirx of RNA,node2vec or one hot ,shape:[RNA_num, identity_feature_dim]
    :param edgelist: ncRNA-protein pairs, columns = [pair_index,RNA_index,protein_index,label]
    :param NODE_INPUT_DIM:identity_feature_dim
    :param SIDE_FEATURE_DIM:side_feature_dim
    :param GCN_HIDDEN_DIM:dimension of embeddings generated by GCN
    :param SIDE_HIDDEN_DIM: dimension of side_feature after fully connected layer transforming
    :param ENCODE_HIDDEN_DIM: the final node representation dimension after combining embeddings and side_feature
    :param threshold: >threshold regard as interaction, otherwise non-interaction
    :param probsavepath: path of saving prediction probability
    :param metricssavepath: path of saving result metrics
    :param use_side_feature:True or False
    :param accumulate_strategy:method of node aggregating (stack or sum)
    :param EPOCHS: training epochs
    :param DROPOUT_RATIO:
    :param INITIAL_LEARNING_RATE: the initial learning rate of training
    :param layers:the number of GCN layers
    :param WEIGHT_DACAY:
    :param step_size: after step_size steps, learning rate will descend
    :param gamma:the percentage of each learning rate droping
    :return:
    '''


    tps = []
    tns = []
    fns = []
    fps = []
    # 随机划分k折交叉验证
    train_data, test_data = train_edgelist.values, test_edgelist.values

    # 训练时计算所有的样本对的得分，但是只有训练集中的通过loss进行优化
    RNA_identity_feature = tensor_from_numpy(RNA_identity_feature, DEVICE).float()
    protein_identity_feature = tensor_from_numpy(protein_identity_feature, DEVICE).float()
    nc_num = RNA_identity_feature.shape[0]
    pr_num = protein_identity_feature.shape[0]


    train_pos_num = len(np.where(train_data[:, 3] == 1)[0])
    train_neg_num = len(np.where(train_data[:, 3] == -1)[0])
    test_pos_num = len(np.where(test_data[:, 3] == 1)[0])
    test_neg_num = len(np.where(test_data[:, 3] == -1)[0])


    print("The number of train data is {},containing positive samples:{} and negative samples:{}"
          .format(len(train_data), train_pos_num, train_neg_num))
    print("The number of valid data is {},containing positive samples:{} and negative samples:{}"
          .format(len(test_data), test_pos_num, test_neg_num))

    # edgelist一共四列：[index,RNA,protein,label]
    # 分别建RNA到protein邻接矩阵的列表和protein到RNA的邻接矩阵列表
    RNA2protein_adj = []
    RNA2protein_adj.append(NPI_neg_matrix)
    RNA2protein_adj.append(NPI_pos_matrix)
    protein2RNA_adj = []
    protein2RNA_adj.append(NPI_neg_matrix.T)
    protein2RNA_adj.append(NPI_pos_matrix.T)
    print("The number of negative samples in the matrix:")
    print(NPI_neg_matrix.sum())
    print("The number of positive samples in the matrix:")
    print(NPI_pos_matrix.sum())
    RNA2protein_adj = globally_normalize_bipartite_adjacency(RNA2protein_adj, False)
    protein2RNA_adj = globally_normalize_bipartite_adjacency(protein2RNA_adj, False)

    # 将numpy.array 转为Tensor
    RNA2protein_adj = [to_torch_sparse_tensor(adj, DEVICE) for adj in RNA2protein_adj]
    protein2RNA_adj = [to_torch_sparse_tensor(adj, DEVICE) for adj in protein2RNA_adj]
    labels = tensor_from_numpy((train_data[:, 3]), DEVICE).long()

    model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                                  SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, use_side_feature=use_side_feature,
                                  accumulate_strategy=accumulate_strategy,
                                  dropout=DROPOUT_RATIO, num_basis=2, layers=layers).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DACAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[80], gamma=0.5, last_epoch=-1)

    # print("初始化的学习率：", optimizer.defaults['lr'])

    for e in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        RNA_indices = train_data[:, 1]
        protein_indices = train_data[:, 2]
        RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
        protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()
        model_inputs = (RNA2protein_adj, protein2RNA_adj,
                        RNA_identity_feature, protein_identity_feature,
                        [], [],
                        RNA_indices, protein_indices,)
        logits = model(*model_inputs)
        prob = F.softmax(logits, dim=1).detach()
        pred_y = torch.sum(prob * SCORES, dim=1).detach()
        y = pred_y.clone().detach()
        y[y > threshold] = 1
        y[y <= threshold] = 0
        loss = criterion(logits, (labels + 1) // 2)
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新

        # print("第%d个epoch的学习率：%f" % (e+1, optimizer.param_groups[0]['lr']))
        scheduler.step()

        printN(y, (labels+1)//2)
        train_acc = accuracy(y, (labels+1)//2)
        train_pre = precision(y, (labels+1)//2)
        train_sen = sensitivity(y, (labels+1)//2)
        train_spe = specificity(y, (labels+1)//2)
        train_MCC = MCC(y, (labels+1)//2)
        train_FPR = FPR(y, (labels+1)//2)
        train_TPR = TPR(y, (labels+1)//2)
        print("Train: accuracy:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, FPR:{}, TPR:{}".
              format(train_acc, train_pre, train_sen, train_spe, train_MCC, train_FPR, train_TPR))
        print('\n')

        #  验证
        if (e + 1) == EPOCHS:
            model.eval()
            with torch.no_grad():
                RNA_indices = test_data[:, 1]
                protein_indices = test_data[:, 2]
                RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
                protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()
                model_inputs = (RNA2protein_adj, protein2RNA_adj,
                                RNA_identity_feature, protein_identity_feature,
                                [], [],
                                RNA_indices, protein_indices,)
                test_logits = model(*model_inputs)
                test_labels = tensor_from_numpy(test_data[:, 3], DEVICE).long()
                test_prob = F.softmax(test_logits, dim=1).detach()
                test_pred_y = torch.sum(test_prob * SCORES, dim=1).detach()
                test_loss = criterion(test_logits, (test_labels + 1) // 2)
                test_y = test_pred_y.clone().detach()
                test_y[test_y > threshold] = 1
                test_y[test_y <= threshold] = 0

                TP, TN, FP, FN = printN(test_y, (test_labels + 1) // 2)
                test_AUC = AUC(test_data[:, 3].squeeze(), test_prob[:, 1].detach().numpy().squeeze())
                test_acc = accuracy(test_y, (test_labels + 1) // 2)
                test_pre = precision(test_y, (test_labels + 1) // 2)
                test_sen = sensitivity(test_y, (test_labels + 1) // 2)
                test_spe = specificity(test_y, (test_labels + 1) // 2)
                test_MCC = MCC(test_y, (test_labels + 1) // 2)
                test_FPR = FPR(test_y, (test_labels + 1) // 2)
                test_TPR = TPR(test_y, (test_labels + 1) // 2)

                print("Test: accuracy:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, FPR:{}, TPR:{}, AUROC:{}".
                      format(test_acc, test_pre, test_sen, test_spe, test_MCC, test_FPR, test_TPR,test_AUC))
                print("==========================")
                print('\n')

            RNA_indices = RNA_indices.numpy()
            protein_indices =  protein_indices.numpy()
            prob0 = test_prob[:,0].numpy()
            prob1 = test_prob[:, 1].numpy()
            pred_score = test_pred_y.numpy()
            pred_label = test_y.numpy()
            true_label = test_labels.numpy()

        model.train()


    res = {'RNA_index': RNA_indices, 'protein_index':protein_indices,'prob0':prob0,
           'prob1':prob1,'pred_score':pred_score,'pred_label':pred_label,'label': true_label}
    res = pd.DataFrame(res)
    res.to_csv(probsavepath,index=False) #保存预测结果

    #final_AUC = AUC(label_list, pred_list)
    final_ACC, final_Sen, final_Spe, final_Pre, final_MCC, final_FPR = test_acc, test_sen, test_spe, test_pre, test_MCC,test_FPR
    print("The final performance of RPI-RGCNAE is:")
    print("ACC: {}, Sen: {}, Spe: {}, Pre: {}, MCC: {}, FPR: {}, AUC: {}".format(final_ACC, final_Sen, final_Spe,
                                                                                 final_Pre, final_MCC, final_FPR,
                                                                                 test_AUC))
    
    sheet = open(metricssavepath, 'a')
    sheet.write(str(final_ACC) + "," + str(final_Sen) + "," + str(final_Spe) + "," + str(final_Pre) + "," + str(
        final_MCC) + "," +
                str(final_FPR) + "," + str(test_AUC) + "\n")
    sheet.close()




def independent_test(filepath, savepath, INI_PATH, DATA_SET, negative_random_sample, layers,
                     with_side_information):
    print("independent_test is runnning")
    print("dataset = {}, negative_random_sample = {}, layers = {}, with_side_information = {}".format(DATA_SET,
                                                                                                      negative_random_sample,
                                                                                                      layers,
                                                                                   with_side_information))
    savepath = os.path.join(savepath,'independent_test')
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    INITIAL_LEARNING_RATE = config.getfloat(DATA_SET, 'INITIAL_LEARNING_RATE')
    WEIGHT_DACAY = config.getfloat(DATA_SET, 'WEIGHT_DACAY')
    DROPOUT_RATIO = config.getfloat(DATA_SET, 'DROPOUT_RATIO')
    step_size = config.getint(DATA_SET, 'step_size')
    gamma = config.getfloat(DATA_SET, 'gamma')
    EPOCHS = config.getint(DATA_SET, 'EPOCHS')
    SIDE_FEATURE_DIM = config.getint(DATA_SET, 'SIDE_FEATURE_DIM')
    GCN_HIDDEN_DIM = config.getint(DATA_SET, 'GCN_HIDDEN_DIM')
    SIDE_HIDDEN_DIM = config.getint(DATA_SET, 'SIDE_HIDDEN_DIM')
    ENCODE_HIDDEN_DIM = config.getint(DATA_SET, 'ENCODE_HIDDEN_DIM')
    WITH_SIDE = "side" if with_side_information else "withoutside"
    for i in range(10):
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        train_edgelist,test_edgelist = \
            load_dataset(dataset=DATA_SET, filepath=filepath, identity_feature_dim=1024,
                         identity_feature='random', negative_random_sample=negative_random_sample)

        save_load_model(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                        train_edgelist,test_edgelist, NODE_INPUT_DIM=1024, SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
                        GCN_HIDDEN_DIM=1024 // (2 ** layers),
                        SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
                        use_side_feature=with_side_information, accumulate_strategy='stack',
                        threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
                        DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers, EPOCHS=EPOCHS,
                        gamma=gamma,
                        probsavepath=os.path.join(savepath,"prob_" +  DATA_SET + "_independent_test" + ".csv"),
                        metricssavepath=os.path.join(savepath,"metrics_" +
                                                     DATA_SET + "_independent_test"+".csv"),
                        embedsavepath='')


if __name__ == "__main__":
    #split_data('NPInter_10412')
    #split_data('RPI7317')

    start = time.time()
    print("start:{}".format(start))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INI_PATH = os.path.join(BASE_DIR, 'dataset_settings.ini')
    filepath = os.path.join(os.path.join(rootPath,'data'),'generated_data')
    #filepath = os.path.join(os.path.join(BASE_DIR, 'data'), 'generated_data')
    savepath = os.path.join(rootPath,'results')
    #savepath = os.path.join(BASE_DIR, 'results')

    parser = argparse.ArgumentParser(
        description="""R-GCN Graph Autoencoder for NcRNA-protein Link Prediciton """)
    parser.add_argument('-method',type = str,help = "choose the method you want to run.",default='independent_test')
    parser.add_argument('-dataset',
                        type=str, help='choose a dataset to implement 5-fold cross validation.', default='NPInter_10412')
    parser.add_argument('-negative_random_sample',
                        type=str, help='choose a method to generate negative samples.',
                        default='sort')
    parser.add_argument('-layers',
                        type=int, default=1)
    parser.add_argument('-with_side_information',type = bool,default=False)

    args = parser.parse_args()
    method = args.method
    DATA_SET = args.dataset
    negative_random_sample = args.negative_random_sample
    layers = args.layers
    with_side_information = args.with_side_information

    if method  == "independent_test":
        filepath = os.path.join(filepath,'independent')
        independent_test(filepath, savepath, INI_PATH, DATA_SET, negative_random_sample, layers, with_side_information)

    end = time.time()
    print("total {} seconds".format(end - start))
