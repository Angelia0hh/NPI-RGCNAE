import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from model2 import *
from utils import *
import os
'''
######hyper
INITIAL_LEARNING_RATE = 0.01
EPOCHS = 100
SIDE_FEATURE_DIM = 343
GCN_HIDDEN_DIM = 512
SIDE_HIDDEN_DIM = 256
ENCODE_HIDDEN_DIM = 256
NUM_BASIS = 2
DROPOUT_RATIO = 0.7
WEIGHT_DACAY = 0.005
######hyper
'''
DEVICE = torch.device('cpu')
SCORES = torch.tensor([-1,1]).to(DEVICE)


def set_seed(seed=1):  
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(dataset, filepath, identity_feature, negative_random_sample):
    if dataset == 'NPInter_10412':
        filepath = filepath +'NPInter_10412/'
    elif dataset == 'RPI13254':
        filepath = filepath +'RPI13254/'
    elif dataset == 'RPI1807':
        filepath = filepath + 'RPI1807/'
    elif dataset == 'RPI488':
        filepath = filepath + 'RPI488/'
    elif dataset == 'RPI369':
        filepath = filepath + 'RPI369/'
    elif dataset == 'RPI7317':
        filepath = filepath + 'RPI7317/'
    elif dataset == 'RPI1446':
        filepath = filepath + 'RPI1446/'
    elif dataset =='RPI2241':
        filepath = filepath + 'RPI2241/'
    elif dataset =='NPInter_4158':
        filepath = filepath + 'NPInter_4158/'
    elif dataset == 'RPI369':
        filepath = filepath + 'RPI369/'
    print(filepath)
    NPI_pos_matrix = pd.read_csv(filepath + 'NPI_pos.csv', header=None).values

    name = ['index']
    for i in range(1024):
        name.append(i + 1)

    if negative_random_sample == 'sort_random':
        NPI_neg_matrix = pd.read_csv(filepath + 'NPI_neg_sort_random.csv', header=None).values
        edgelist = pd.read_csv(filepath + 'edgelist_sort_random.csv', header=None)
        emb = pd.read_csv(filepath + 'emd_sort_random.emd.txt', header=None, sep=' ', names=name)
    elif negative_random_sample == 'sort':
        NPI_neg_matrix = pd.read_csv(filepath + 'NPI_neg_sort.csv', header=None).values
        edgelist = pd.read_csv(filepath + 'edgelist_sort.csv', header=None)
        emb = pd.read_csv(filepath + 'emd_sort.emd.txt', header=None, sep=' ', names=name)
    elif negative_random_sample == 'random':
        NPI_neg_matrix = pd.read_csv(filepath + 'NPI_neg_random.csv', header=None).values
        edgelist = pd.read_csv(filepath + 'edgelist_random.csv', header=None)
        emb = pd.read_csv(filepath + 'emd_random.emd.txt', header=None, sep=' ', names=name)
    elif negative_random_sample == 'raw':
        NPI_neg_matrix = pd.read_csv(filepath + 'NPI_neg_raw.csv', header=None).values
        edgelist = pd.read_csv(filepath + 'edgelist_raw.csv', header=None)
        emb = pd.read_csv(filepath + 'emd_raw.emd.txt', header=None, sep=' ', names=name)

    protein_side_feature = pd.read_csv(filepath + 'Protein3merfeat.csv').values
    RNA_side_feature = pd.read_csv(filepath + 'ncRNA4merfeat.csv').values
    supplement = np.zeros((RNA_side_feature.shape[0], 87))  # 通过补零补齐到同一维度
    RNA_side_feature = np.concatenate((RNA_side_feature, supplement), axis=1)


    if (identity_feature == 'one hot'):
        identity_feature = np.identity(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], dtype=np.float32)
        RNA_identity_feature, protein_identity_feature = identity_feature[
                                                         :NPI_pos_matrix.shape[0]], identity_feature[
                                                                                    NPI_pos_matrix.shape[0]:]
    elif (identity_feature == 'node2vec'):
        emb.sort_values('index', inplace=True)  # 按index排序
        emb = emb[list(range(1, 1025))].values
        RNA_identity_feature, protein_identity_feature = emb[:NPI_pos_matrix.shape[0]], emb[NPI_pos_matrix.shape[0]:]
    elif (identity_feature =='random'):
        feature = np.random.randn(NPI_pos_matrix.shape[0]+NPI_pos_matrix.shape[1],1024)
        RNA_identity_feature, protein_identity_feature = feature[:NPI_pos_matrix.shape[0]], feature[NPI_pos_matrix.shape[0]:]
    elif (identity_feature == 'kmer'):
        RNA_identity_feature = RNA_side_feature
        protein_identity_feature = protein_side_feature
    return NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, \
           RNA_identity_feature, protein_side_feature, RNA_side_feature, edgelist


def train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,protein_side_feature, RNA_side_feature, edgelist,
          NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM, SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM,
          threshold, probsavepath, metricssavepath,embedsavepath, use_side_feature, accumulate_strategy,
          EPOCHS=100, DROPOUT_RATIO = 0.7, INITIAL_LEARNING_RATE = 0.01, layers = 1, WEIGHT_DACAY = 0.005, step_size = 10, gamma = 0.7):
    '''

    :param NPI_pos_matrix: binary matrix of positive samples，shape:[ncRNA_num,protein_num]
    :param NPI_neg_matrix: binary matrix of negative samples， shape:[ncRNA_num, protein_num]
    :param protein_identity_feature: identity feature matirx of proteins, node2vec or one hot, shape:[protein_num, identity_feature_dim]
    :param RNA_identity_feature: identity feature matirx of RNA,node2vec or one hot ,shape:[RNA_num, identity_feature_dim]
    :param protein_side_feature: side_feature matirx of protein, 3mer, shape:[protein_num,343]
    :param RNA_side_feature: side_feature matirx of RNA, 4mer, shape:[RNA_num,256]
    :param edgelist: ncRNA-protein paris, columns = [pair_index,RNA_index,protein_index,label]
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

    valid_acc = []
    valid_pre= []
    valid_sen = []
    valid_spe = []
    valid_MCC = []
    valid_FPR = []
    valid_TPR = []
    valid_loss = []
    valid_RMSE = []
    valid_AUC = []
    pred_list = []
    label_list = []
    # 随机划分k折交叉验证
    train_data, test_data = get_k_fold_data(5, edgelist)

    # 训练时计算所有的样本对的得分，但是只有训练集中的通过loss进行优化
    RNA_indices = edgelist.values[:, 1]
    protein_indices = edgelist.values[:, 2]
    RNA_side_feature = tensor_from_numpy(RNA_side_feature, DEVICE).float()
    protein_side_feature = tensor_from_numpy(protein_side_feature, DEVICE).float()
    RNA_identity_feature = tensor_from_numpy(RNA_identity_feature, DEVICE).float()
    protein_identity_feature = tensor_from_numpy(protein_identity_feature, DEVICE).float()
    RNA_indices = tensor_from_numpy(RNA_indices, DEVICE).long()
    protein_indices = tensor_from_numpy(protein_indices, DEVICE).long()
    nc_num = RNA_side_feature.shape[0]
    pr_num = protein_side_feature.shape[0]


    sheet = open(metricssavepath, 'a')
    sheet.write(" " + "," + 'acc' + "," + 'pre' + "," + 'sen' + "," + 'spe' + "," + 'MCC' + "," +
                'FPR' + "," + 'TPR' + "," + 'AUC' + "\n")
    sheet.close()


    for i in range(5):

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
        print(mask.sum())
        print(NPI_pos_matrix.sum())
        print(NPI_neg_matrix.sum())
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
                                      SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, use_side_feature = use_side_feature,accumulate_strategy = accumulate_strategy,
                                      dropout=DROPOUT_RATIO, num_basis=2, layers=layers).to(DEVICE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DACAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= step_size, gamma= gamma, last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[80], gamma=0.5, last_epoch=-1)

        print("初始化的学习率：", optimizer.defaults['lr'])

        for e in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            logits,_,_ = model(*model_inputs)
            prob = F.softmax(logits, dim=1).detach()
            pred_y = torch.sum(prob * SCORES, dim=1).detach()
            print("The {}th training,the whole number of train dataset is {}:".format(e+1,len(train_mask)))
            print("probability:")
            print(prob)
            print("prediction:")
            y = pred_y.clone().detach()
            y[y > threshold] = 1
            y[y <=threshold] = 0
            print(y[train_mask])
            print('score:')
            print(pred_y[train_mask])
            print("label:")
            print(labels)
            loss = criterion(logits[train_mask], (labels+1)//2)
            rmse = expected_rmse(logits[train_mask], labels)

            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            print("第%d个epoch的学习率：%f" % (e+1, optimizer.param_groups[0]['lr']))
            scheduler.step()

            print("Epoch {:03d}: Loss: {:.4f}, RMSE: {:.4f}".format(e, loss.item(), rmse.item()))
            printN(y[train_mask], (labels+1)//2)
            train_acc = accuracy(y[train_mask], (labels+1)//2)
            train_pre = precision(y[train_mask], (labels+1)//2)
            train_sen = sensitivity(y[train_mask], (labels+1)//2)
            train_spe = specificity(y[train_mask], (labels+1)//2)
            train_MCC = MCC(y[train_mask], (labels+1)//2)
            train_FPR = FPR(y[train_mask], (labels+1)//2)
            train_TPR = TPR(y[train_mask], (labels+1)//2)
            print("accuracy:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, FPR:{}, TPR:{}".
                  format(train_acc, train_pre, train_sen, train_spe, train_MCC, train_FPR, train_TPR))
            print('\n')

            #  验证
            if (e + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_logits, embedding_protein, embedding_RNA = model(*model_inputs)
                    test_labels = tensor_from_numpy(test_data[i][:, 3], DEVICE).long()
                    test_mask = tensor_from_numpy(test_data[i][:, 0], DEVICE)
                    test_prob = F.softmax(test_logits, dim=1).detach()
                    test_pred_y = torch.sum(test_prob * SCORES, dim=1).detach()
                    test_loss = criterion(test_logits[test_mask], (test_labels+1)//2)
                    rmse = expected_rmse(test_logits[test_mask], test_labels)
                    print("==========================")
                    print('validation')
                    print("probability:")
                    print(test_prob)
                    print("prediction:")
                    test_y = test_pred_y.clone().detach()
                    test_y[test_y > threshold] = 1
                    test_y[test_y <=threshold] = 0
                    print(test_y[test_mask])
                    print("score:")
                    print(test_pred_y[test_mask])
                    print("label:")
                    print(test_labels)
                    print('Test On Epoch {}: loss: {:.4f}, Test rmse: {:.4f}'.format(e, test_loss.item(), rmse.item()))

                    printN(test_y[test_mask], (test_labels+1)//2)
                    test_AUC = AUC(test_data[i][:, 3].squeeze(), test_pred_y[test_mask].detach().numpy().squeeze())
                    test_acc = accuracy(test_y[test_mask], (test_labels+1)//2)
                    test_pre = precision(test_y[test_mask], (test_labels+1)//2)
                    test_sen = sensitivity(test_y[test_mask], (test_labels+1)//2)
                    test_spe = specificity(test_y[test_mask], (test_labels+1)//2)
                    test_MCC = MCC(test_y[test_mask], (test_labels+1)//2)
                    test_FPR = FPR(test_y[test_mask], (test_labels+1)//2)
                    test_TPR = TPR(test_y[test_mask], (test_labels+1)//2)
                    print("accuracy:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, FPR:{}, TPR:{}, AUC:{}".
                          format(test_acc, test_pre, test_sen, test_spe, test_MCC ,test_FPR, test_TPR, test_AUC))
                    print("==========================")
                    print('\n')


                if (e+1)==EPOCHS:

                    sheet  = open(metricssavepath,'a')
                    sheet.write("fold"+str(i) + "," +str(test_acc)+","+str(test_pre)+","+str(test_sen)+","+str(test_spe)+","+str(test_MCC)+","+
                                str(test_FPR)+","+str(test_TPR)+","+str(test_AUC)+"\n")
                    sheet.close()

                    if(i==4):
                        embedding_RNA = pd.DataFrame(embedding_RNA)
                        embedding_protein = pd.DataFrame(embedding_protein)
                        #embedding_protein.to_csv(embedsavepath,header=False,index=False)
                        #embedding_RNA.to_csv(embedsavepath, header=False,index=False)
                    valid_acc.append(test_acc)
                    valid_pre.append(test_pre)
                    valid_sen.append(test_sen)
                    valid_spe.append(test_spe)
                    valid_MCC.append(test_MCC)
                    valid_FPR.append(test_FPR)
                    valid_TPR.append(test_TPR)
                    valid_AUC.append(test_AUC)
                    valid_loss.append(test_loss)
                    valid_RMSE.append(rmse)
                model.train()

        if len(test_pred_y[test_mask]) == (len(edgelist)//5):
            pred_list.append(test_pred_y[test_mask].numpy())
            label_list.append(test_labels.numpy())
        else:
            pred_list.append(test_pred_y[test_mask].numpy()[:len(edgelist)//5])
            label_list.append(test_labels.numpy()[:len(edgelist)//5])

    pred_res = np.array(pred_list)
    label_res = np.array(label_list)
    res = np.concatenate([pred_res,label_res],axis=0)

    names = []
    [names.append('pred ' + str(i + 1)) for i in range(5)]
    [names.append('label ' + str(i + 1)) for i in range(5)]
    res = pd.DataFrame(res.T,columns = names)
    res.to_csv(probsavepath,index=False)


    print("average accuracy:"+str(np.mean(valid_acc)))
    print(valid_acc)
    print("average precision:" + str(np.mean(valid_pre)))
    print(valid_pre)
    print("average sensitivity:" + str(np.mean(valid_sen)))
    print(valid_sen)
    print("average specificity:" + str(np.mean(valid_spe)))
    print(valid_spe)
    print("average MCC:" + str(np.mean(valid_MCC)))
    print(valid_MCC)
    print("average FPR:" + str(np.mean(valid_FPR)))
    print(valid_FPR)
    print("average TPR:" + str(np.mean(valid_TPR)))
    print(valid_TPR)
    print("average AUC:"+str(np.mean(valid_AUC)))
    print(valid_AUC)

    sheet = open(metricssavepath, 'a')
    sheet.write(
        str('average'+','+str(np.mean(valid_acc))) + "," + str(np.mean(valid_pre)) + "," + str(np.mean(valid_sen)) + "," + str(np.mean(valid_spe)) + "," + str(np.mean(valid_MCC)) + "," +
        str(np.mean(valid_FPR)) + "," + str(np.mean(valid_TPR)) + "," + str(np.mean(valid_AUC)) + "\n")
    sheet.close()


def expected_rmse(logits, label):
    prob = F.softmax(logits, dim=1)
    pred_y = torch.sum(prob * SCORES, dim=1)
    diff = torch.pow(label - pred_y, 2)

    return torch.sqrt(diff.mean())


if __name__ == "__main__":
    set_seed(1)
    filepath = 'data/generated_data/'
    savepath = 'result/'


    INI_PATH = 'dataset_settings.ini'
    parser = argparse.ArgumentParser(
        description="""R-GCN Graph Autoencoder for NcRNA-protein Link Prediciton """)

    parser.add_argument('-dataset',
                        type=str, help='choose a dataset to implement 5-fold cross validation', default='NPInter_10412')

    args = parser.parse_args()
    #DATA_SET = args.dataset


    '''
    if dataset == 'NPInter_10412':
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        protein_side_feature, RNA_side_feature, edgelist = \
            load_dataset(dataset= dataset, filepath=filepath,
                         identity_feature='node2vec', negative_random_sample='raw')

        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.5, step_size=10,
              gamma=0.7,
              savepath=filepath)
    elif dataset =='RPI1446':
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        protein_side_feature, RNA_side_feature, edgelist = \
            load_dataset(dataset=dataset, filepath=filepath,
                         identity_feature='node2vec', negative_random_sample='raw')

        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.5, step_size=10,
              gamma=0.7,
              savepath=filepath)

    elif dataset == 'RPI2241':
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        protein_side_feature, RNA_side_feature, edgelist = \
            load_dataset(dataset='RPI2241', filepath=filepath,
                         identity_feature='node2vec', negative_random_sample='sort_random')

        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=True, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=0.001, WEIGHT_DACAY=0.009, DROPOUT_RATIO=0., step_size=10,
              gamma=0.5,
              metricssavepath=" ",probsavepath=" ")

    elif dataset =='NPInter_4158':
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        protein_side_feature, RNA_side_feature, edgelist = \
            load_dataset(dataset='NPInter_4158', filepath=filepath,
                         identity_feature='node2vec', negative_random_sample='sort_random')

        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0., DROPOUT_RATIO=0., step_size=20,
              gamma=0.7,
              probsavepath=" ",metricssavepath=" ")

    elif dataset == ' ':
        # NPInter10412数据集不同组合下的结果
        negative_sample_method = ['random', 'sort_random', 'sort']
        accumulate_strategy = ['stack', 'sum']
        node_feature_type = ['node2vec','one hot']
        for i in negative_sample_method:
            for j in accumulate_strategy:
                for k in node_feature_type:
                    if k == 'node2vec':
                        NODE_INPUT_DIM = 1024
                    elif k == 'one hot':
                        NODE_INPUT_DIM = 5085
                    # node2vec/one hot +side_information
                    name = i+"_"+k+"_"+j
                    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                    protein_side_feature, RNA_side_feature, edgelist = \
                        load_dataset(dataset='NPInter_10412', filepath=filepath,
                                     identity_feature=k, negative_random_sample=i)

                    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                          protein_side_feature,
                          RNA_side_feature, edgelist, NODE_INPUT_DIM=NODE_INPUT_DIM, use_side_feature=True, accumulate_strategy=j,
                          threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.5, step_size=10,
                          gamma=0.7,
                          probsavepath=filepath + 'NPInter_10412/result/predictions/'+'prob_'+name + '_side.csv',
                          metricssavepath= filepath + 'NPInter_10412/result/metrics/'+name + "_side.csv")

                    # node2vec/one hot
                    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
                    protein_side_feature, RNA_side_feature, edgelist = \
                        load_dataset(dataset='NPInter_10412', filepath=filepath,
                                     identity_feature=k, negative_random_sample=i)

                    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
                          protein_side_feature,
                          RNA_side_feature, edgelist, NODE_INPUT_DIM=NODE_INPUT_DIM, use_side_feature=False, accumulate_strategy=j,
                          threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.5, step_size=10,
                          gamma=0.7,
                          probsavepath=filepath + 'NPInter_10412/result/predictions/'+'prob_'+name + '.csv',
                          metricssavepath=filepath + 'NPInter_10412/result/metrics/'+name + ".csv")
    elif dataset == 'RPI7317':
        name = 'RPI7317_'
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        protein_side_feature, RNA_side_feature, edgelist = \
            load_dataset(dataset='RPI7317', filepath=filepath,
                         identity_feature='node2vec', negative_random_sample='sort_random')

        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
              protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.5, step_size=10,
              gamma=0.7,
              probsavepath=filepath + 'NPInter_10412/result/predictions/' + 'prob_' + name + '.csv',
              metricssavepath=filepath + 'NPInter_10412/result/metrics/' + name + ".csv")

    elif dataset =='RPI369':
        name = 'RPI369'
        NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
        protein_side_feature, RNA_side_feature, edgelist = \
            load_dataset(dataset='RPI369', filepath=filepath,
                         identity_feature='node2vec', negative_random_sample='sort')

        train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,
              protein_side_feature,
              RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
              threshold=0, INITIAL_LEARNING_RATE=0.001, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.3, step_size=10,
              gamma=0.7,
              probsavepath=filepath + 'NPInter_10412/result/predictions/' + 'prob_' + name + '.csv',
              metricssavepath=filepath + 'NPInter_10412/result/metrics/' + name + ".csv")
        
    '''


    '''
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset = 'NPInter_10412', filepath = 'C:/Users/yuhan/Desktop/data/generated_data/',
                     identity_feature = 'node2vec', negative_random_sample = 'sort_random')
    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,protein_side_feature, RNA_side_feature, edgelist,
          threshold=0,INITIAL_LEARNING_RATE = 0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.7, step_size=10, gamma=0.7,
          savepath='C:/Users/yuhan/Desktop/data/generated_data/NPInter10412/result/NPInter10412_sort_random_node2vec_stack.csv')
    
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset='RPI1807', filepath='C:/Users/yuhan/Desktop/data/generated_data/',
                     identity_feature='node2vec', negative_random_sample='None')
    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature,protein_side_feature, RNA_side_feature, edgelist,
          threshold=0, INITIAL_LEARNING_RATE = 0.01, WEIGHT_DACAY = 0.005, DROPOUT_RATIO = 0.7, step_size=10, gamma=0.7)
    
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset='RPI1446', filepath='C:/Users/yuhan/Desktop/data/generated_data/',
                     identity_feature='one hot', negative_random_sample='sort_random')

    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
          RNA_side_feature, edgelist,
          threshold=0, INITIAL_LEARNING_RATE = 0.01, WEIGHT_DACAY=0.009, DROPOUT_RATIO=0.7, step_size=10, gamma=0.7,
          savepath='C:/Users/yuhan/Desktop/data/generated_data/RPI1446/result/RPI13254_sort_random_node2vec_stack.csv')
    '''
    '''
    # NPInter10412数据集不同组合下的结果
    negative_sample_method = ['random','sort_random','sort']
    accumulate_strategy = ['stack','sum']
    for i in negative_sample_method:
        for j in accumulate_strategy:
            # node2vec+side_information
            NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
            protein_side_feature, RNA_side_feature, edgelist = \
                load_dataset(dataset='NPInter_10412', filepath=filepath,
                             identity_feature='node2vec', negative_random_sample=i)

            train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
                  RNA_side_feature, edgelist, NODE_INPUT_DIM=1024,use_side_feature=True, accumulate_strategy=j,
                  threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.7, step_size=10, gamma=0.7,
                  savepath=savepath+'prob_NPInter10412_node2vec_side_'+i+"_"+j+'.csv')

            # one hot +side_information
            NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
            protein_side_feature, RNA_side_feature, edgelist = \
                load_dataset(dataset='NPInter_10412', filepath=filepath,
                             identity_feature='one hot', negative_random_sample=i)

            train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
                  RNA_side_feature, edgelist, NODE_INPUT_DIM=5085, use_side_feature=True, accumulate_strategy=j,
                  threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.7, step_size=10,
                  gamma=0.7,
                  savepath=savepath+'prob_NPInter10412_one hot_side_'+i+"_"+j+'.csv')
            # one hot
            NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
            protein_side_feature, RNA_side_feature, edgelist = \
                load_dataset(dataset='NPInter_10412', filepath=filepath,
                             identity_feature='one hot', negative_random_sample=i)

            train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
                  RNA_side_feature, edgelist,NODE_INPUT_DIM=5085,use_side_feature=False, accumulate_strategy=j,
                  threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.7, step_size=10,
                  gamma=0.7,
                  savepath=savepath + 'prob_NPInter10412_one hot_' + i + "_" + j + '.csv')
            # node2vec
            NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
            protein_side_feature, RNA_side_feature, edgelist = \
                load_dataset(dataset='NPInter_10412', filepath=filepath,
                             identity_feature='node2vec', negative_random_sample=i)

            train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
                  RNA_side_feature, edgelist,NODE_INPUT_DIM=1024,use_side_feature=False, accumulate_strategy=j,
                  threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.7, step_size=10,
                  gamma=0.7,
                  savepath=savepath + 'prob_NPInter10412_node2vec_' + i + "_" + j + '.csv')
    
    # 1807数据库结果
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset='RPI1446', filepath=filepath,
                     identity_feature='node2vec', negative_random_sample='raw')

    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
          RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
          threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.009, DROPOUT_RATIO=0.7, step_size=10,
          gamma=0.7,
          savepath=savepath)
          
          
          
              

    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset='NPInter_4158', filepath=filepath,
                     identity_feature='node2vec', negative_random_sample='sort_random')

    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
          RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=False, accumulate_strategy='stack',
          threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0., DROPOUT_RATIO=0., step_size=20,
          gamma=0.7,
          savepath=savepath)
    
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset='RPI13254', filepath=filepath,
                     identity_feature='node2vec', negative_random_sample='sort_random')

    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
          RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, use_side_feature=True, accumulate_strategy='stack',
          threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.005, DROPOUT_RATIO=0.5, step_size=20,
          gamma=0.7,
          savepath=savepath)
    '''


    for DATA_SET in ['RPI369', 'RPI7317','RPI13254','RPI2241', 'NPInter_10412'][0:1]:
        config = configparser.ConfigParser()
        config.read(INI_PATH)
        
        INITIAL_LEARNING_RATE = config.getfloat(DATA_SET,'INITIAL_LEARNING_RATE')
        WEIGHT_DACAY = config.getfloat(DATA_SET,'WEIGHT_DACAY')
        DROPOUT_RATIO = config.getfloat(DATA_SET,'DROPOUT_RATIO')
        step_size = config.getint(DATA_SET,'step_size')
        gamma = config.getfloat(DATA_SET,'gamma')
        layers = config.getint(DATA_SET,'layers')
        EPOCHS = config.getint(DATA_SET,'EPOCHS')
        SIDE_FEATURE_DIM = config.getint(DATA_SET,'SIDE_FEATURE_DIM')
        GCN_HIDDEN_DIM = config.getint(DATA_SET,'GCN_HIDDEN_DIM')
        SIDE_HIDDEN_DIM = config.getint(DATA_SET,'SIDE_HIDDEN_DIM')
        ENCODE_HIDDEN_DIM = config.getint(DATA_SET,'ENCODE_HIDDEN_DIM')
        for negative_generation in ['sort','sort_random','random']:
            NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
            protein_side_feature, RNA_side_feature, edgelist = \
                load_dataset(dataset=DATA_SET, filepath=filepath,
                             identity_feature='node2vec', negative_random_sample=negative_generation)

            train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
                  RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, SIDE_FEATURE_DIM=SIDE_FEATURE_DIM, GCN_HIDDEN_DIM=GCN_HIDDEN_DIM,
                  SIDE_HIDDEN_DIM=SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM=ENCODE_HIDDEN_DIM,
                  use_side_feature=True, accumulate_strategy='stack',
                  threshold=0, INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE, WEIGHT_DACAY=WEIGHT_DACAY, DROPOUT_RATIO=DROPOUT_RATIO, step_size=step_size, layers=layers,
                  gamma=gamma, probsavepath=savepath + DATA_SET + "/prob_"+negative_generation+"_stack_node2vec_side.csv",
                  metricssavepath=savepath + DATA_SET + "/metrics_"+negative_generation+"_stack_node2vec_side.csv",
                  embedsavepath = savepath+DATA_SET+"/embedding_protein.csv")
    
    '''
    NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, \
    protein_side_feature, RNA_side_feature, edgelist = \
        load_dataset(dataset='NPInter_10412', filepath=filepath,
                     identity_feature='node2vec', negative_random_sample='sort')

    train(NPI_pos_matrix, NPI_neg_matrix, protein_identity_feature, RNA_identity_feature, protein_side_feature,
          RNA_side_feature, edgelist, NODE_INPUT_DIM=1024, SIDE_FEATURE_DIM=343, GCN_HIDDEN_DIM=512,
          SIDE_HIDDEN_DIM=256, ENCODE_HIDDEN_DIM=256,
          use_side_feature=True, accumulate_strategy='stack',
          threshold=0, INITIAL_LEARNING_RATE=0.01, WEIGHT_DACAY=0.003, DROPOUT_RATIO=0.5, step_size=10, layers=1,
          gamma=0.7, probsavepath=savepath + 'NPInter_10412' + "/prob_sort_stack_node2vec_side.csv",
          metricssavepath=savepath + 'NPInter_10412' + "/metrics_sort_stack_node2vec_side.csv")
    
    '''







