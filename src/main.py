import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import time
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from model import*
from utils import*

DEVICE = torch.device('cpu')
SCORES = torch.tensor([-1, 1]).to(DEVICE)
'''
seed = 1
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''


def load_dataset(dataset, filepath, identity_feature, negative_random_sample, identity_feature_dim=1024, use_side_feature = False):

    filepath = os.path.join(filepath,dataset)
    NPI_pos_matrix = pd.read_csv(os.path.join(filepath,'NPI_pos.csv'), header=None).values

    name = ['index']
    for i in range(1024):
        name.append(i + 1)

    NPI_neg_matrix = pd.read_csv(os.path.join(filepath,"NPI_neg_" + negative_random_sample + ".csv"), header=None).values
    edgelist = pd.read_csv(os.path.join(filepath,'edgelist_' + negative_random_sample + '.csv'), header=None)
    protein_side_feature = []
    RNA_side_feature = []

    if use_side_feature:
        protein_side_feature = pd.read_csv( os.path.join(filepath, 'Protein3merfeat.csv')).values
        RNA_side_feature = pd.read_csv(os.path.join(filepath,'ncRNA4merfeat.csv')).values
        supplement = np.zeros((RNA_side_feature.shape[0], 87))  # 通过补零补齐到同一维度
        RNA_side_feature = np.concatenate((RNA_side_feature, supplement), axis=1)

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
           RNA_identity_feature, protein_side_feature, RNA_side_feature, edgelist


def loadData(dataset, filepath, identity_feature, partition, identity_feature_dim, use_side_information = False):
    filepath = os.path.join(filepath, dataset)
    if "_0." not in dataset:
        NPI_pos_matrix = pd.read_csv(os.path.join(filepath, 'NPI_pos.csv'), header=None).values
        NPI_neg_matrix = pd.read_csv(os.path.join(filepath, "NPI_neg_" + negative_random_sample + ".csv"),
                                     header=None).values
        edgelist = pd.read_csv(os.path.join(filepath, 'edgelist_' + negative_random_sample + '.csv'), header=None)
        protein_side_feature = []
        RNA_side_feature = []
        if use_side_information:
            protein_side_feature = pd.read_csv(os.path.join(filepath, 'Protein3merfeat.csv')).values
            RNA_side_feature = pd.read_csv(os.path.join(filepath, 'ncRNA4merfeat.csv')).values
            supplement = np.zeros((RNA_side_feature.shape[0], 87))  # 通过补零补齐到同一维度
            RNA_side_feature = np.concatenate((RNA_side_feature, supplement), axis=1)
    else:
        NPI_pos_matrix = pd.read_csv(os.path.join(filepath, 'NPI_pos'+str(partition)+'.csv'), header=None).values
        NPI_neg_matrix = pd.read_csv(os.path.join(filepath, "NPI_neg" +str(partition) + ".csv"),
                                     header=None).values
        edgelist = pd.read_csv(os.path.join(filepath, 'edgelist' + str(partition) + '.csv'), header=None)
        protein_side_feature = []
        RNA_side_feature = []
        if use_side_information:
            protein_side_feature = pd.read_csv(os.path.join(filepath, 'protein3merfeat'+str(partition)+'.csv')).values
            RNA_side_feature = pd.read_csv(os.path.join(filepath, 'rna4merfeat'+str(partition)+'.csv')).values
            supplement = np.zeros((RNA_side_feature.shape[0], 87))  # 通过补零补齐到同一维度
            RNA_side_feature = np.concatenate((RNA_side_feature, supplement), axis=1)
    name = ['index']
    for i in range(1024):
        name.append(i + 1)



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
           RNA_identity_feature, protein_side_feature, RNA_side_feature, edgelist

def train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
          RNA_side_feature, edgelist,
          NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM, SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM,
          threshold, probsavepath, metricssavepath, embedsavepath, use_side_feature, accumulate_strategy,
          EPOCHS, DROPOUT_RATIO, INITIAL_LEARNING_RATE, layers, WEIGHT_DACAY, step_size, gamma):
    '''

    :param NPI_pos_matrix: binary matrix of positive samples，shape:[ncRNA_num,protein_num]
    :param NPI_neg_matrix: binary matrix of negative samples， shape:[ncRNA_num, protein_num]
    :param protein_identity_feature: identity feature matirx of proteins, node2vec or one hot, shape:[protein_num, identity_feature_dim]
    :param RNA_identity_feature: identity feature matirx of RNA,node2vec or one hot ,shape:[RNA_num, identity_feature_dim]
    :param protein_side_feature: side_feature matirx of protein, 3mer, shape:[protein_num,343]
    :param RNA_side_feature: side_feature matirx of RNA, 4mer, shape:[RNA_num,256]
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

    pred_list = []
    label_list = []
    tps = []
    tns = []
    fns = []
    fps = []
    RNA_embs = []
    protein_embs = []

    # 随机划分k折交叉验证
    train_data, test_data = get_k_fold_data(5, edgelist)

    # 训练时计算所有的样本对的得分，但是只有训练集中的通过loss进行优化
    if use_side_feature:
        RNA_side_feature = tensor_from_numpy(RNA_side_feature, DEVICE).float()
        protein_side_feature = tensor_from_numpy(protein_side_feature, DEVICE).float()
    RNA_identity_feature = tensor_from_numpy(RNA_identity_feature, DEVICE).float()
    protein_identity_feature = tensor_from_numpy(protein_identity_feature, DEVICE).float()
    nc_num = RNA_identity_feature.shape[0]
    pr_num = protein_identity_feature.shape[0]

    for i in range(5):
        plot_train_loss = []
        plot_valid_loss = []
        plot_train_acc = []
        plot_valid_acc = []

        train_pos_num = len(np.where(train_data[i][:, 3] == 1)[0])
        train_neg_num = len(np.where(train_data[i][:, 3] == -1)[0])
        test_pos_num = len(np.where(test_data[i][:, 3] == 1)[0])
        test_neg_num = len(np.where(test_data[i][:, 3] == -1)[0])

        print("This is the {}th cross validation ".format(i + 1))
        print("The number of train data is {},containing positive samples:{} and negative samples:{}"
              .format(len(train_data[i]), train_pos_num, train_neg_num))
        print("The number of valid data is {},containing positive samples:{} and negative samples:{}"
              .format(len(test_data[i]), test_pos_num, test_neg_num))

        # edgelist一共四列：[index,RNA,protein,label]
        mask = np.ones((nc_num, pr_num))
        mask[test_data[i][:, 1], test_data[i][:, 2]] = 0  # 将验证集中的边置0
        print(mask.sum())
        print(NPI_pos_matrix.sum())
        print(NPI_neg_matrix.sum())
        tmp_pos = NPI_pos_matrix * mask
        tmp_neg = NPI_neg_matrix * mask

        # 分别建RNA到protein邻接矩阵的列表和protein到RNA的邻接矩阵列表
        RNA2protein_adj = []
        RNA2protein_adj.append(tmp_neg)
        RNA2protein_adj.append(tmp_pos)
        protein2RNA_adj = []
        protein2RNA_adj.append(tmp_neg.T)
        protein2RNA_adj.append(tmp_pos.T)
        print("The number of negative samples in the matrix:")
        print(tmp_neg.sum())
        print("The number of positive samples in the matrix:")
        print(tmp_pos.sum())
        RNA2protein_adj = globally_normalize_bipartite_adjacency(RNA2protein_adj, False)
        protein2RNA_adj = globally_normalize_bipartite_adjacency(protein2RNA_adj, False)

        # 将numpy.array 转为Tensor
        RNA2protein_adj = [to_torch_sparse_tensor(adj, DEVICE) for adj in RNA2protein_adj]
        protein2RNA_adj = [to_torch_sparse_tensor(adj, DEVICE) for adj in protein2RNA_adj]
        labels = tensor_from_numpy((train_data[i][:, 3]), DEVICE).long()
        train_mask = tensor_from_numpy(train_data[i][:, 0], DEVICE)

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
            RNA_indices = train_data[i][:, 1]
            protein_indices = train_data[i][:, 2]
            RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
            protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()
            model_inputs = (RNA2protein_adj, protein2RNA_adj,
                            RNA_identity_feature, protein_identity_feature,
                            RNA_side_feature, protein_side_feature,
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

            '''
            printN(y, (labels+1)//2)
            train_acc = accuracy(y, (labels+1)//2)
            train_pre = precision(y, (labels+1)//2)
            train_sen = sensitivity(y, (labels+1)//2)
            train_spe = specificity(y, (labels+1)//2)
            train_MCC = MCC(y, (labels+1)//2)
            train_FPR = FPR(y, (labels+1)//2)
            train_TPR = TPR(y, (labels+1)//2)
            print("accuracy:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, FPR:{}, TPR:{}".
                  format(train_acc, train_pre, train_sen, train_spe, train_MCC, train_FPR, train_TPR))
            print('\n')
            plot_train_loss.append(loss)
            plot_train_acc.append(train_acc)
            '''

            #  验证
            model.eval()
            with torch.no_grad():
                RNA_indices = test_data[i][:, 1]
                protein_indices = test_data[i][:, 2]
                RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
                protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()
                model_inputs = (RNA2protein_adj, protein2RNA_adj,
                                RNA_identity_feature, protein_identity_feature,
                                RNA_side_feature, protein_side_feature,
                                RNA_indices, protein_indices,)
                test_logits = model(*model_inputs)
                test_labels = tensor_from_numpy(test_data[i][:, 3], DEVICE).long()
                test_prob = F.softmax(test_logits, dim=1).detach()
                test_pred_y = torch.sum(test_prob * SCORES, dim=1).detach()
                test_loss = criterion(test_logits, (test_labels + 1) // 2)
                print("==========================")
                print("Epoch {:03d}: Loss: {:.4f}".format(e, test_loss.item()))
                print('validation')

                test_y = test_pred_y.clone().detach()
                test_y[test_y > threshold] = 1
                test_y[test_y <= threshold] = 0


                TP, TN, FP, FN = printN(test_y, (test_labels + 1) // 2)
                test_AUC = AUC(test_data[i][:, 3].squeeze(), test_prob[:,1].detach().numpy().squeeze())
                test_acc = accuracy(test_y, (test_labels + 1) // 2)
                test_pre = precision(test_y, (test_labels + 1) // 2)
                test_sen = sensitivity(test_y, (test_labels + 1) // 2)
                test_spe = specificity(test_y, (test_labels + 1) // 2)
                test_MCC = MCC(test_y, (test_labels + 1) // 2)
                test_FPR = FPR(test_y, (test_labels + 1) // 2)
                test_TPR = TPR(test_y, (test_labels + 1) // 2)

                print("accuracy:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, FPR:{}, TPR:{}, AUC:{}".
                      format(test_acc, test_pre, test_sen, test_spe, test_MCC, test_FPR, test_TPR, test_AUC))
                print("==========================")
                print('\n')

                if (e + 1) == EPOCHS:
                    tps.append(TP)
                    fps.append(FP)
                    tns.append(TN)
                    fns.append(FN)

                    pred_list.append(test_prob[:,1].numpy())
                    label_list.append(test_labels.numpy())

            model.train()

    pred_list = np.concatenate(pred_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    res = {'pred': pred_list, 'label': label_list}
    res = pd.DataFrame(res)
    res.to_csv(probsavepath,index=False) #保存预测结果

    final_AUC = AUC(label_list, pred_list)
    final_ACC, final_Sen, final_Spe, final_Pre, final_MCC, final_FPR = performance(tps, tns, fps, fns)
    print("The final performance of RPI-RGCNAE is:")
    print("ACC: {}, Sen: {}, Spe: {}, Pre: {}, MCC: {}, FPR: {}, AUC: {}".format(final_ACC, final_Sen, final_Spe,
                                                                                 final_Pre, final_MCC, final_FPR,
                                                                                 final_AUC))

    sheet = open(metricssavepath, 'a')
    sheet.write(str(final_ACC) + "," + str(final_Sen) + "," + str(final_Spe) + "," + str(final_Pre) + "," + str(
        final_MCC) + "," +
                str(final_FPR) + "," + str(final_AUC) + "\n")
    sheet.close()


def compare_different_achitectures(filepath, savepath, INI_PATH):
    print("compare_different_achitectures is running")
    # 比较不同的网络结构
    for DATA_SET in ['NPInter_10412', 'RPI369','RPI2241','RPI7317']:
        config = configparser.ConfigParser()
        config.read(INI_PATH)

        INITIAL_LEARNING_RATE = config.getfloat(DATA_SET, 'INITIAL_LEARNING_RATE')
        WEIGHT_DACAY = config.getfloat(DATA_SET, 'WEIGHT_DACAY')
        DROPOUT_RATIO = config.getfloat(DATA_SET, 'DROPOUT_RATIO')
        step_size = config.getint(DATA_SET, 'step_size')
        gamma = config.getfloat(DATA_SET, 'gamma')
        EPOCHS = config.getint(DATA_SET, 'EPOCHS')
        SIDE_FEATURE_DIM = config.getint(DATA_SET, 'SIDE_FEATURE_DIM')

        structure = {'NODE_INPUT_DIM': [128, 256, 512, 1024],
                     'GCN_HIDDEN_DIM': [64, 128, 256, 512],
                     'SIDE_HIDDEN_DIM': [32, 64, 128, 256],
                     'ENCODE_HIDDEN_DIM': [32, 64, 128, 256]}

        for c in range(4):
            s = str(structure['NODE_INPUT_DIM'][c]) + "_" + str(structure['GCN_HIDDEN_DIM'][c]) + "_" + str(
                structure['SIDE_HIDDEN_DIM'][c]) + "_" + str(structure['ENCODE_HIDDEN_DIM'][c])
            for i in range(10):
                NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                protein_side_feature, RNA_side_feature, edgelist = \
                    load_dataset(dataset=DATA_SET, filepath=filepath,
                                 identity_feature_dim=structure['NODE_INPUT_DIM'][c],
                                 identity_feature='random', negative_random_sample='sort')

                train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                      protein_side_feature,
                      RNA_side_feature, edgelist, NODE_INPUT_DIM=structure['NODE_INPUT_DIM'][c],
                      SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
                      GCN_HIDDEN_DIM=structure['GCN_HIDDEN_DIM'][c],
                      SIDE_HIDDEN_DIM=structure['SIDE_HIDDEN_DIM'][c],
                      ENCODE_HIDDEN_DIM=structure['ENCODE_HIDDEN_DIM'][c],
                      use_side_feature=False, accumulate_strategy='stack',
                      threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
                      DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=1, EPOCHS=EPOCHS,
                      gamma=gamma,
                      probsavepath=savepath +os.sep +DATA_SET +os.sep +"parameter"+os.sep+"prob_" + s + "_sort_stack_random_withoutside.csv",
                      metricssavepath=savepath +os.sep +DATA_SET +os.sep +"parameter"+os.sep+"metrics_" + s + "_sort_stack_random_withoutside.csv",
                      embedsavepath='')


def compare_different_combinations(filepath, savepath, INI_PATH):
    print("compare_different_combinations is running")
    # 不同特征对比组合
    #for DATA_SET in ['NPInter_10412', 'RPI369']:
    for DATA_SET in ['RPI7317','RPI2241','NPInter_10412', 'RPI369']:
        config = configparser.ConfigParser()
        config.read(INI_PATH)

        INITIAL_LEARNING_RATE = config.getfloat(DATA_SET, 'INITIAL_LEARNING_RATE')
        WEIGHT_DACAY = config.getfloat(DATA_SET, 'WEIGHT_DACAY')
        DROPOUT_RATIO = config.getfloat(DATA_SET, 'DROPOUT_RATIO')
        step_size = config.getint(DATA_SET, 'step_size')
        gamma = config.getfloat(DATA_SET, 'gamma')
        layers = config.getint(DATA_SET, 'layers')
        EPOCHS = config.getint(DATA_SET, 'EPOCHS')
        SIDE_FEATURE_DIM = config.getint(DATA_SET, 'SIDE_FEATURE_DIM')
        GCN_HIDDEN_DIM = config.getint(DATA_SET, 'GCN_HIDDEN_DIM')
        SIDE_HIDDEN_DIM = config.getint(DATA_SET, 'SIDE_HIDDEN_DIM')
        ENCODE_HIDDEN_DIM = config.getint(DATA_SET, 'ENCODE_HIDDEN_DIM')

        negative_sample_method = ['sort']
        accumulate_strategy = ['stack']
        #node_feature_type = ['random', 'one hot']
        node_feature_type = ['random']
        for i in negative_sample_method:
            for j in accumulate_strategy:
                for k in node_feature_type:
                    if k == 'random':
                        NODE_INPUT_DIM = 1024
                    elif k == 'one hot':
                        if DATA_SET == 'RPI369':
                            NODE_INPUT_DIM = 669
                        elif DATA_SET == 'NPInter_10412':
                            NODE_INPUT_DIM = 5085
                    name = i + "_" + k + "_" + j
                    for t in range(10):
                        # random/one hot +side_information
                        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                        protein_side_feature, RNA_side_feature, edgelist = \
                            load_dataset(dataset=DATA_SET, filepath=filepath, identity_feature_dim=NODE_INPUT_DIM,
                                         identity_feature=k, negative_random_sample=i)

                        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                              protein_side_feature,
                              RNA_side_feature, edgelist, NODE_INPUT_DIM=NODE_INPUT_DIM,
                              SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
                              GCN_HIDDEN_DIM=GCN_HIDDEN_DIM, SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM,
                              ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
                              use_side_feature=True, accumulate_strategy=j,
                              threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
                              DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers, EPOCHS=EPOCHS,
                              gamma=gamma,
                              probsavepath=savepath+os.sep+ DATA_SET +os.sep+ "combination"+os.sep+"prob_" + name + "_side.csv",
                              metricssavepath=savepath +os.sep+ DATA_SET +os.sep+ "combination"+os.sep+"metrics_" + name + "_side.csv",
                              embedsavepath='')

                        # random/one hot
                        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                        protein_side_feature, RNA_side_feature, edgelist = \
                            load_dataset(dataset=DATA_SET, filepath=filepath, identity_feature_dim=NODE_INPUT_DIM,
                                         identity_feature=k, negative_random_sample=i)

                        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                              protein_side_feature,
                              RNA_side_feature, edgelist, NODE_INPUT_DIM=NODE_INPUT_DIM,
                              SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
                              GCN_HIDDEN_DIM=GCN_HIDDEN_DIM, SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM,
                              ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
                              use_side_feature=False, accumulate_strategy=j,
                              threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
                              DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers, EPOCHS=EPOCHS,
                              gamma=gamma,
                              probsavepath=savepath +os.sep+ DATA_SET +os.sep +"combination"+os.sep+"prob_" + name + "_withoutside.csv",
                              metricssavepath=savepath + os.sep+DATA_SET + os.sep+"combination"+os.sep+"metrics_" + name + "_withoutside.csv",
                              embedsavepath='')


def compare_different_layers(filepath, savepath, INI_PATH):
    print("compare_different_layers is running")
    # 比较不同的层数
    #for DATA_SET in ['NPInter_10412', 'RPI369']:
    for DATA_SET in ['RPI7317', 'RPI2241','NPInter_10412', 'RPI369']:
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

        for l in range(1, 5):
            for i in range(10):
                NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                protein_side_feature, RNA_side_feature, edgelist = \
                    load_dataset(dataset=DATA_SET, filepath=filepath,
                                 identity_feature_dim=1024,
                                 identity_feature='random', negative_random_sample='sort')

                train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                      protein_side_feature,
                      RNA_side_feature, edgelist, NODE_INPUT_DIM=1024,
                      SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
                      GCN_HIDDEN_DIM=1024//(2**l),
                      SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM,
                      ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
                      use_side_feature=False, accumulate_strategy='stack',
                      threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
                      DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=l, EPOCHS=EPOCHS,
                      gamma=gamma,
                      probsavepath=savepath +os.sep +DATA_SET +os.sep+ "parameter"+os.sep+"prob_" + str(
                          l) + " layers_sort_stack_random_withoutside.csv",
                      metricssavepath=savepath +os.sep +DATA_SET +os.sep+ "parameter"+os.sep+"metrics_" + str(
                          l) + " layers_sort_stack_random_withoutside.csv",
                      embedsavepath='')


def compare_negative_sample_methods(filepath, savepath, INI_PATH):
    print("compare_negative_sample_methods is runnning")
    # 不同数据集上负样本生成方法的对比
    for DATA_SET in ['NPInter_10412', 'RPI7317', 'RPI2241', 'RPI369']:
        config = configparser.ConfigParser()
        config.read(INI_PATH)

        for negative_generation in ['sort', 'random', 'sort random']:
            if negative_generation == 'random' and DATA_SET =='RPI369':
                param_name = 'RPI369 random'
            else:
                param_name = DATA_SET
            INITIAL_LEARNING_RATE = config.getfloat(param_name, 'INITIAL_LEARNING_RATE')
            WEIGHT_DACAY = config.getfloat(param_name, 'WEIGHT_DACAY')
            DROPOUT_RATIO = config.getfloat(param_name, 'DROPOUT_RATIO')
            step_size = config.getint(param_name, 'step_size')
            gamma = config.getfloat(param_name, 'gamma')
            layers = config.getint(param_name, 'layers')
            EPOCHS = config.getint(param_name, 'EPOCHS')
            SIDE_FEATURE_DIM = config.getint(param_name, 'SIDE_FEATURE_DIM')
            GCN_HIDDEN_DIM = config.getint(param_name, 'GCN_HIDDEN_DIM')
            SIDE_HIDDEN_DIM = config.getint(param_name, 'SIDE_HIDDEN_DIM')
            ENCODE_HIDDEN_DIM = config.getint(param_name, 'ENCODE_HIDDEN_DIM')
            for i in range(10):
                if negative_generation == 'sort random':
                    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                    protein_side_feature, RNA_side_feature, edgelist = \
                        load_dataset(dataset=DATA_SET, filepath=filepath, identity_feature_dim=1024,
                                     identity_feature='random', negative_random_sample='sort_random')
                else:
                    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                    protein_side_feature, RNA_side_feature, edgelist = \
                        load_dataset(dataset=DATA_SET, filepath=filepath, identity_feature_dim=1024,
                                     identity_feature='random', negative_random_sample=negative_generation)

                train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                      protein_side_feature,
                      RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
                      GCN_HIDDEN_DIM=GCN_HIDDEN_DIM,
                      SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
                      use_side_feature=False, accumulate_strategy='stack',
                      threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
                      DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers, EPOCHS=EPOCHS,
                      gamma=gamma,
                      probsavepath=savepath +os.sep+ DATA_SET + os.sep+"negative_method"+os.sep+"prob_" + negative_generation + "_stack_random_withoutside.csv",
                      metricssavepath=savepath +os.sep +DATA_SET +os.sep +"negative_method"+os.sep+"metrics_" + negative_generation + "_stack_random_withoutside.csv",
                      embedsavepath=savepath + DATA_SET + '')


def single_dataset_prediction(filepath, savepath, INI_PATH, DATA_SET, negative_random_sample, layers, with_side_information):
    print("single_dataset_prediction is runnning")
    print("dataset = {}, negative_random_sample = {}, layers = {}, with_side_information = {}".format(DATA_SET,
                                                                                                      negative_random_sample,
                                                                                                      layers,
                                                                                                      with_side_information))
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
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset=DATA_SET, filepath=filepath, identity_feature_dim=1024,
                     identity_feature='random', negative_random_sample=negative_random_sample)
    for i in range(10):
        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
              protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
              GCN_HIDDEN_DIM=1024 // (2 ** layers),
              SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
              use_side_feature=with_side_information, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
              DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers, EPOCHS=EPOCHS,
              gamma=gamma,
              probsavepath= os.path.join(savepath,DATA_SET + os.sep+"single_prediction"+os.sep+"prob_" + negative_random_sample + "_stack_random_" + WITH_SIDE + "_" + str(layers) + ".csv"),
              metricssavepath=os.path.join(savepath, DATA_SET +os.sep +"single_prediction"+os.sep+"metrics_" + negative_random_sample + "_stack_random_" + WITH_SIDE + "_" + str(layers) + ".csv"),
              embedsavepath='')
        print("\n")


def timeAnalysis(filepath, savepath, INI_PATH, DATA_SET, negative_random_sample, layers,
                              with_side_information):
    print("timeAnalysis is runnning")
    print("dataset = {}, negative_random_sample = {}, layers = {}, with_side_information = {}".format(DATA_SET,
                                                                                                      negative_random_sample,
                                                                                                      layers,
                                                                                                      with_side_information))
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    INITIAL_LEARNING_RATE = config.getfloat(DATA_SET, 'INITIAL_LEARNING_RATE')
    WEIGHT_DACAY = config.getfloat(DATA_SET, 'WEIGHT_DACAY')
    DROPOUT_RATIO = config.getfloat(DATA_SET, 'DROPOUT_RATIO')
    step_size = config.getint(DATA_SET, 'step_size')
    gamma = config.getfloat(DATA_SET, 'gamma')
    EPOCHS = config.getint(DATA_SET, 'EPOCHS')
    SIDE_FEATURE_DIM = config.getint(DATA_SET, 'SIDE_FEATURE_DIM')
    #GCN_HIDDEN_DIM = config.getint(DATA_SET, 'GCN_HIDDEN_DIM')
    SIDE_HIDDEN_DIM = config.getint(DATA_SET, 'SIDE_HIDDEN_DIM')
    ENCODE_HIDDEN_DIM = config.getint(DATA_SET, 'ENCODE_HIDDEN_DIM')
    WITH_SIDE = "side" if with_side_information else "withoutside"
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        loadData(dataset=DATA_SET, filepath=filepath, identity_feature='random', partition = negative_random_sample, identity_feature_dim=1024)
    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
          protein_side_feature,
          RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, SIDE_FEATURE_DIM=SIDE_FEATURE_DIM,
          GCN_HIDDEN_DIM=1024 // (2 ** layers),
          SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
          use_side_feature=with_side_information, accumulate_strategy='stack',
          threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY,
          DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers, EPOCHS=EPOCHS,
          gamma=gamma,
          probsavepath=os.path.join(savepath,
                                    DATA_SET + os.sep + "prob_" + negative_random_sample + "_stack_random_" + WITH_SIDE + "_" + str(
                                        layers) + ".csv"),
          metricssavepath=os.path.join(savepath,
                                       DATA_SET + os.sep + "metrics_" + negative_random_sample + "_stack_random_" + WITH_SIDE + "_" + str(
                                           layers) + ".csv"),
          embedsavepath='')


if __name__ == "__main__":
    start = time.time()
    print("start:{}".format(start))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INI_PATH = os.path.join(BASE_DIR, 'dataset_settings.ini')
    filepath = os.path.join(os.path.join(rootPath,'data'),'generated_data')
    savepath = os.path.join(rootPath,'results')

    parser = argparse.ArgumentParser(
        description="""R-GCN Graph Autoencoder for NcRNA-protein Link Prediciton """)
    parser.add_argument('-method',type = str,help = "choose the method you want to run.",default='single_dataset_prediction')
    parser.add_argument('-dataset',
                        type=str, help='choose a dataset to implement 5-fold cross validation.', default='RPI369')
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

    if method == "compare_different_combinations":
        compare_different_combinations(filepath, savepath, INI_PATH)
    elif method  == "compare_different_layers":
        compare_different_layers(filepath, savepath, INI_PATH)
    elif method == "compare_negative_sample_methods":
        compare_negative_sample_methods(filepath, savepath, INI_PATH)
    elif method  == "single_dataset_prediction":
        single_dataset_prediction(filepath, savepath, INI_PATH, DATA_SET, negative_random_sample, layers, with_side_information)
    elif method == 'timeAnalysis':
        timeAnalysis(filepath, savepath, INI_PATH, DATA_SET, negative_random_sample, 1, False)
    end = time.time()
    print("total {} seconds".format(end - start))


