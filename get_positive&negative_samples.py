import numpy as np
import pandas as pd
import math
import random

np.random.seed(1)
random.seed(1)

def calculate_protein_sw_similarity(pr1, pr2, swscore_matrix):
    '''计算每对蛋白质之间规范化后的SmithWaterman similarity'''
    score = swscore_matrix[pr1, pr2]/math.sqrt(swscore_matrix[pr1, pr1]*swscore_matrix[pr2, pr2])
    return score


def calculate_socre_of_pri_and_RNAj(pr_i, RNA_j,positive_samples,swscore_matrix):
    '''
    计算蛋白质i和RNAj之间的互作得分
    '''
    score = 0
    related_pair = [pair for pair in positive_samples if pair[0] == RNA_j]
    for pair in related_pair:
        if(pair[1]!= pr_i):
            score += calculate_protein_sw_similarity(pr_i,pair[1],swscore_matrix)
    return score


def get_positive_samples_of_NPInter(NPI_filepath):
    NPInter = pd.read_table(NPI_filepath)
    protein = NPInter['UNIPROT-ID'].unique().tolist()  # 蛋白质列表
    ncRNA = NPInter['NONCODE-ID'].unique().tolist()  # RNA列表
    positive_index = []  # 相互作用对的下标[ncRNA_index,protein_index,label]
    for index, row in NPInter.iterrows():
        i = ncRNA.index(row['NONCODE-ID'])
        j = protein.index(row['UNIPROT-ID'])
        positive_index.append([i, j])
    return positive_index, protein, ncRNA


def get_Positives_and_Negatives (positive_samples, pr_list, RNA_list,swscore_matrix, savepath):
    Positives = []
    Negatives = []

    # random pair
    for RNA_index in range((len(RNA_list))):
        for pr_index in range(len(pr_list)):
            sample = [RNA_index, pr_index]
            if [RNA_index, pr_index] in positive_samples:
                Ms = 1
                sample.append(Ms)
                Positives.append(sample)
            else:
                Ms = calculate_socre_of_pri_and_RNAj(pr_index, RNA_index, positive_samples,swscore_matrix)
                sample.append(Ms)
                Negatives.append(sample)
    Negatives = sorted(Negatives, key=lambda x: x[2])

    Positives = pd.DataFrame(Positives, columns=['RNA', 'protein', 'label'])
    Negatives = pd.DataFrame(Negatives, columns=['RNA', 'protein', 'label'])
    Positives.to_csv(savepath+'Positives.csv', index=False)
    Negatives.to_csv(savepath+'Negatives.csv', index=False)

    return Positives, Negatives


def get_edgelist(Positives, Negatives, method, savepath, nc_num):
    '''

    :param Positives: 带header的DataFrame，['RNA', 'protein', 'label'] ,label = 1
    :param Negatives: 带header的DataFrame，['RNA', 'protein', 'label']，所有的负样本，分数升序排列
    :param nc_num: RNA的数量
    :return:
    '''
    if method == 'sort':
        Negatives = Negatives[:(len(Positives))]  # 按score升序选取和正样本一样多的负样本
    elif method == 'random':
        Negatives = Negatives.sample(n=len(Positives), random_state=1)  # 随机抽取负样本
    elif method == 'sort_random':
        # 先按照分数取正样本两倍的负样本，再在其中抽取等量正样本
        Negatives = Negatives[:len(Positives) * 2]
        Negatives = Negatives.sample(n=len(Positives), random_state=1)
    elif method =='raw':
        # 原数据集自带的负样本
        pass
    Negatives.loc[:, 'label'] = -1

    # 按升序选取的负样本生成对应的节点对、正负样本矩阵
    edgelist = [Positives, Negatives]
    edgelist = pd.concat(edgelist, axis=0)
    edgelist = edgelist.take(np.random.permutation(len(edgelist))) # 打乱正负样本的顺序
    edgelist = edgelist.reset_index(drop=True)

    if method == 'sort':
        edgelist.to_csv(savepath + 'edgelist_sort.csv', header=None)
        edgelist['protein'] = edgelist['protein'] + nc_num
        edgelist = edgelist[['RNA', 'protein']]
        np.savetxt(savepath + 'graph.edgelist_sort.txt', edgelist, fmt='%s',
               delimiter=' ')
    elif method == 'random':
        edgelist.to_csv(savepath + 'edgelist_random.csv', header=None)
        edgelist['protein'] = edgelist['protein'] + nc_num
        edgelist = edgelist[['RNA', 'protein']]
        np.savetxt(savepath + 'graph.edgelist_random.txt', edgelist, fmt='%s',
                   delimiter=' ')
    elif method == 'sort_random':
        edgelist.to_csv(savepath + 'edgelist_sort_random.csv', header=None)
        edgelist['protein'] = edgelist['protein'] + nc_num
        edgelist = edgelist[['RNA', 'protein']]
        np.savetxt(savepath + 'graph.edgelist_sort_random.txt', edgelist, fmt='%s',
                   delimiter=' ')
    elif method == 'raw':
        edgelist.to_csv(savepath + 'edgelist_raw.csv', header=None)
        edgelist['protein'] = edgelist['protein'] + nc_num
        edgelist = edgelist[['RNA', 'protein']]
        np.savetxt(savepath + 'graph.edgelist_raw.txt', edgelist, fmt='%s',
                   delimiter=' ')

    return Positives, Negatives


def get_NPInter(filepath, savepath):
    '''

    :param filepath: 原始数据的路径
    :param savepath: 中间生成的数据的存储路径
    :return:
    '''
    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    positive_samples, protein, ncRNA = get_positive_samples_of_NPInter(filepath)
    #Positives,Negatives = get_Positives_and_Negatives(positive_samples, protein, ncRNA, swscore_matrix, savepath)
    Positives = pd.read_csv(savepath + 'Positives.csv')
    Negatives = pd.read_csv(savepath + 'Negatives.csv')  #所有的负样本对

    Positives, Negatives_sort = get_edgelist(Positives, Negatives,'sort',savepath, len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_random = get_edgelist(Positives, Negatives,'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_sort, NPI_neg_random ,NPI_neg_sort_random  =  np.zeros((len(ncRNA), len(protein))),  np.zeros((len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))
    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath+'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)


def get_RPI13254(filepath, savepath):

    RPI13254_positive = pd.read_table(filepath + 'RPI13254_positive.txt')
    RPI13254_negative = pd.read_table(filepath + 'RPI13254_negative.txt')

    RPI13254_pos = RPI13254_positive['gene'].tolist()
    protein_pos = [item[:7] for item in RPI13254_pos]  # 正样本中的蛋白质那一列（包含重复）
    ncRNA_pos = [item[8:] for item in RPI13254_pos]  # 正样本中RNA那列（包含重复）

    RPI13254_neg = RPI13254_negative['gene'].tolist()
    protein_neg = [item[:7] for item in RPI13254_neg]  # 负样本中蛋白质列（包含重复）
    ncRNA_neg = [item[8:] for item in RPI13254_neg]  # 负样本中RNA列（包含重复）

    protein = list(set(protein_pos).union(set(protein_neg)))  # 蛋白质去重后对应的列表
    ncRNA = list(set(ncRNA_pos).union(set(ncRNA_neg)))  # RNA去重后对应的列表

    merge1 = {"RNA": ncRNA_pos,"protein": protein_pos}
    RPI13254_pos = pd.DataFrame(merge1)  # 正样本对 [RNA,protein] 是每对的名称而不是下标
    merge2 = {"RNA": ncRNA_neg, "protein": protein_neg}
    RPI13254_neg = pd.DataFrame(merge2)

    # 去掉没有对应的sequence的RNA
    discard_list = ['YBL039W-A', 'YBL101W-A', 'YFL057C', 'YIR044C', 'YAR062W', 'YNL097C-A']
    index_list = RPI13254_pos[RPI13254_pos['RNA'].isin(discard_list)].index
    RPI13254_pos = RPI13254_pos.drop(index=index_list)
    ncRNA = RPI13254_pos['RNA'].unique().tolist()
    protein = RPI13254_pos['protein'].unique().tolist()
    print("RPI13254:")
    print("protein:" + str(len(protein)) + " RNA:" + str(len(ncRNA)) + " positives:" + str(
        len(RPI13254_pos)) + " negatives:" + str(len(RPI13254_neg)))

    print(ncRNA)
    print(protein)

    positive_index = []
    negative_index = []

    for index, row in RPI13254_pos.iterrows():
        i = ncRNA.index(row['RNA'])
        j = protein.index(row['protein'])
        positive_index.append([i, j])

    for index, row in RPI13254_neg.iterrows():
        i = ncRNA.index(row['RNA'])
        j = protein.index(row['protein'])
        negative_index.append([i, j, -1])

    # 原数据集自带的负样本，是每对RNA-蛋白质的index
    Negatives_raw = pd.DataFrame(negative_index,columns = ['RNA', 'protein', 'label'])
    Negatives_raw.to_csv(savepath+'Negatives_raw.csv', index=False)

    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    Positives, Negatives = get_Positives_and_Negatives(positive_index, protein, ncRNA, swscore_matrix, savepath)
    Positives = pd.read_csv(savepath + 'Positives.csv')
    Negatives = pd.read_csv(savepath + 'Negatives.csv')

    Positives, Negatives_sort = get_edgelist(Positives, Negatives, 'sort', savepath,
                                             len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_raw = get_edgelist(Positives, Negatives_raw, 'raw', savepath, len(ncRNA))
    _, Negatives_random = get_edgelist(Positives, Negatives, 'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_sort, NPI_neg_raw, NPI_neg_random, NPI_neg_sort_random = np.zeros((len(ncRNA), len(protein))), np.zeros(
        (len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))

    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath + 'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_raw[Negatives_raw.values[:, 0], Negatives_raw.values[:, 1]] = 1
    NPI_neg_raw = pd.DataFrame(NPI_neg_raw)
    NPI_neg_raw.to_csv(savepath + 'NPI_neg_raw.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)


def get_RPI7317(filepath, savepath):
    RPI7317 = pd.read_csv(filepath)
    protein = RPI7317['Protein names'].unique().tolist()
    ncRNA = RPI7317['RNA names'].unique().tolist()
    positive_index = []
    print(protein)
    print(ncRNA)
    for index, row in RPI7317.iterrows():
        i = ncRNA.index(row['RNA names'])
        j = protein.index(row['Protein names'])
        positive_index.append([i, j])

    print("RPI7317")
    print("protein:" + str(len(protein)) + " RNA:" + str(len(ncRNA)) + " positives:" + str(len(positive_index)))
    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    # Positives, Negatives = get_Positives_and_Negatives(positive_index, protein, ncRNA, swscore_matrix,savepath)
    Positives = pd.read_csv(savepath + 'Positives.csv')
    Negatives = pd.read_csv(savepath + 'Negatives.csv')

    Positives, Negatives_sort = get_edgelist(Positives, Negatives, 'sort', savepath,
                                             len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_random = get_edgelist(Positives, Negatives, 'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_sort, NPI_neg_random, NPI_neg_sort_random = np.zeros((len(ncRNA), len(protein))), np.zeros(
        (len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))
    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath + 'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)


def get_RPI1446(filepath, savepath):
    RPI1446 = pd.read_table(filepath, header=None, names=['protein', 'RNA','label'])
    protein = RPI1446['protein'].unique().tolist()
    ncRNA = RPI1446['RNA'].unique().tolist()
    positive_index = []
    negative_index = []

    for index,row in RPI1446.iterrows():
        i = ncRNA.index(row['RNA'])
        j = protein.index(row['protein'])
        if(row['label']==1):
            positive_index.append([i,j])
        else:
            negative_index.append([i,j,-1])
    print("RPI1446:")
    print("positive:"+str(len(positive_index))+" negative:"+str(len(negative_index))+" RNA:"+str(len(ncRNA))+" protein:"+str(len(protein)))
    Negatives_raw = pd.DataFrame(negative_index, columns=['RNA', 'protein', 'label'])
    Negatives_raw.to_csv(savepath + 'Negatives_raw.csv', index=False)
    Positives = pd.DataFrame(positive_index, columns =['RNA', 'protein'])
    Positives['label'] = 1
    get_edgelist(Positives, Negatives_raw, 'raw', savepath, len(ncRNA))

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_raw = np.zeros((len(ncRNA), len(protein)))
    NPI_neg_raw[Negatives_raw.values[:, 0], Negatives_raw.values[:, 1]] = 1
    NPI_neg_raw = pd.DataFrame(NPI_neg_raw)
    NPI_neg_raw.to_csv(savepath + 'NPI_neg_raw.csv', index=False, header=None)


def get_RPI1807(filepath, savepath):
    RPI1807_positive = pd.read_table(filepath + 'RPI1807_PositivePairs.csv')
    RPI1807_negative = pd.read_table(filepath + 'RPI1807_NegativePairs.csv')
    protein = list(set(RPI1807_positive['Protein ID'].tolist()).union(set(RPI1807_negative['Protein ID'].tolist())))
    ncRNA = list(set(RPI1807_positive['RNA ID'].tolist()).union(set(RPI1807_negative['RNA ID'].tolist())))
    protein_pos = RPI1807_positive['Protein ID'].unique().tolist()
    ncRNA_pos = RPI1807_positive['RNA ID'].unique().tolist()
    positive_index = []  # 相互作用对的下标[ncRNA_index,protein_index,label]
    negative_index = []

    for index, row in RPI1807_positive.iterrows():
        i = ncRNA.index(row['RNA ID'])
        j = protein.index(row['Protein ID'])
        positive_index.append([i, j])

    for index, row in RPI1807_negative.iterrows():
        i = ncRNA.index(row['RNA ID'])
        j = protein.index(row['Protein ID'])
        negative_index.append([i, j, -1])

    Positives = pd.DataFrame(positive_index, columns = ['RNA','protein'])
    Positives.loc[:,'label'] = 1
    Negatives_raw = pd.DataFrame(negative_index,columns = ['RNA','protein','label'])
    Positives.to_csv(savepath +'Positives.csv',index=False)
    Negatives_raw.to_csv(savepath + 'Negatives_raw.csv',index=False)
    _, Negatives_raw = get_edgelist(Positives, Negatives_raw, 'raw', savepath, len(ncRNA))
    print("RPI1807")
    print("protein:" + str(len(protein)) + " RNA:" + str(len(ncRNA)) + " positives:" + str(len(positive_index)))

    positive_index = []
    for index, row in RPI1807_positive.iterrows():
        i = ncRNA_pos.index(row['RNA ID'])
        j = protein_pos.index(row['Protein ID'])
        positive_index.append([i, j])

    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    Positives, Negatives = get_Positives_and_Negatives(positive_index, protein, ncRNA, swscore_matrix, savepath)

    Positives, Negatives_sort = get_edgelist(Positives, Negatives, 'sort', savepath,
                                             len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_random = get_edgelist(Positives, Negatives, 'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_pos, NPI_neg_random, NPI_neg_sort, NPI_neg_sort_random, NPI_neg_raw = np.zeros((len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein))),\
                                                                              np.zeros((len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))
    NPI_pos [Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath + 'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_raw[Negatives_raw.values[:, 0], Negatives_raw.values[:, 1]] = 1
    NPI_neg_raw = pd.DataFrame(NPI_neg_raw)
    NPI_neg_raw.to_csv(savepath + 'NPI_neg_raw.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)


def get_RPI369(filepath, savepath):
    RPI369 = pd.read_table(filepath , header=None,
                           names=['protein', 'RNA', 'label'])
    RPI369_pos = RPI369[RPI369['label']==1]
    protein = RPI369_pos['protein'].unique().tolist()
    ncRNA = RPI369_pos['RNA'].unique().tolist()
    positive_index = []  # 相互作用对的下标[ncRNA_index,protein_index,label]

    for index, row in RPI369_pos.iterrows():
        i = ncRNA.index(row['RNA'])
        j = protein.index(row['protein'])
        positive_index.append([i, j])
    print("RPI369:")
    print("positive:" + str(len(positive_index)) + " RNA:" + str(len(ncRNA)) + " protein:" + str(len(protein)))


    Positives = pd.DataFrame(positive_index, columns=['RNA', 'protein'])
    Positives['label'] = 1

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    Positives,Negatives = get_Positives_and_Negatives(positive_index, protein, ncRNA, swscore_matrix, savepath)


    Positives, Negatives_sort = get_edgelist(Positives, Negatives, 'sort', savepath,
                                             len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_random = get_edgelist(Positives, Negatives, 'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_neg_sort, NPI_neg_random, NPI_neg_sort_random = np.zeros((len(ncRNA), len(protein))), np.zeros(
        (len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))
    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath + 'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)


def get_RPI2241(filepath, savepath):
    RPI2241 = pd.read_table(filepath + 'RPI2241_pairs.txt', header=None, names=['Protein', 'RNA', 'label'])
    protein = RPI2241['Protein'].unique().tolist()
    ncRNA = RPI2241['RNA'].unique().tolist()
    positive_index = []
    negative_index = []

    for index, row in RPI2241.iterrows():
        i = ncRNA.index(row['RNA'])
        j = protein.index(row['Protein'])
        if (row['label'] == 1):
            positive_index.append([i, j])
        else:
            negative_index.append([i, j, -1])
    print("RPI2241:")
    print("positive:" + str(len(positive_index)) + " negative:" + str(len(negative_index)) + " RNA:" + str(
        len(ncRNA)) + " protein:" + str(len(protein)))
    Negatives_raw = pd.DataFrame(negative_index, columns=['RNA', 'protein', 'label'])
    Negatives_raw.to_csv(savepath + 'Negatives_raw.csv', index=False)

    Positives = pd.DataFrame(positive_index, columns=['RNA', 'protein'])
    Positives['label'] = 1
    get_edgelist(Positives, Negatives_raw, 'raw', savepath, len(ncRNA))

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_raw = np.zeros((len(ncRNA), len(protein)))
    NPI_neg_raw[Negatives_raw.values[:, 0], Negatives_raw.values[:, 1]] = 1
    NPI_neg_raw = pd.DataFrame(NPI_neg_raw)
    NPI_neg_raw.to_csv(savepath + 'NPI_neg_raw.csv', index=False, header=None)

    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    Positives,Negatives = get_Positives_and_Negatives(positive_index, protein, ncRNA, swscore_matrix, savepath)


    Positives, Negatives_sort = get_edgelist(Positives, Negatives, 'sort', savepath,
                                             len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_random = get_edgelist(Positives, Negatives, 'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_neg_sort, NPI_neg_random, NPI_neg_sort_random = np.zeros((len(ncRNA), len(protein))), np.zeros(
        (len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))
    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath + 'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)


def get_NPInter4158(filepath, savepath):
    NPInter4158 = pd.read_table(filepath + 'NPInter4158_interaction.txt', header=None, sep=' ')
    ncRNA = pd.read_table(filepath + 'NPInter4158_lncRNA.txt', header=None)
    protein = pd.read_table(filepath + 'NPInter4158_protein.txt', header=None)
    NPInter4158 = NPInter4158.values
    index = NPInter4158.nonzero()
    positive_index = []
    for i, _ in enumerate(index[0]):
        positive_index.append([index[0][i], index[1][i]])

    swpath = savepath + 'protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values

    Positives,Negatives = get_Positives_and_Negatives(positive_index, protein, ncRNA, swscore_matrix, savepath)
    # Positives = pd.read_csv(savepath + 'Positives.csv')
    # Negatives = pd.read_csv(savepath + 'Negatives.csv')  # 所有的负样本对

    Positives, Negatives_sort = get_edgelist(Positives, Negatives, 'sort', savepath,
                                             len(ncRNA))  # 得到合并前的正负样本对，dataframe
    _, Negatives_random = get_edgelist(Positives, Negatives, 'random', savepath, len(ncRNA))
    _, Negatives_sort_random = get_edgelist(Positives, Negatives, 'sort_random', savepath, len(ncRNA))

    NPI_pos = np.zeros((len(ncRNA), len(protein)))
    NPI_pos[Positives.values[:, 0], Positives.values[:, 1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_pos.to_csv(savepath + 'NPI_pos.csv', index=False, header=None)

    NPI_neg_sort, NPI_neg_random, NPI_neg_sort_random = np.zeros((len(ncRNA), len(protein))), np.zeros(
        (len(ncRNA), len(protein))), np.zeros((len(ncRNA), len(protein)))
    NPI_neg_sort[Negatives_sort.values[:, 0], Negatives_sort.values[:, 1]] = 1
    NPI_neg_sort = pd.DataFrame(NPI_neg_sort)
    NPI_neg_sort.to_csv(savepath + 'NPI_neg_sort.csv', index=False, header=None)

    NPI_neg_random[Negatives_random.values[:, 0], Negatives_random.values[:, 1]] = 1
    NPI_neg_random = pd.DataFrame(NPI_neg_random)
    NPI_neg_random.to_csv(savepath + 'NPI_neg_random.csv', index=False, header=None)

    NPI_neg_sort_random[Negatives_sort_random.values[:, 0], Negatives_sort_random.values[:, 1]] = 1
    NPI_neg_sort_random = pd.DataFrame(NPI_neg_sort_random)
    NPI_neg_sort_random.to_csv(savepath + 'NPI_neg_sort_random.csv', index=False, header=None)

if __name__ == '__main__':
    # NPInter10412
    # get_NPInter('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\NPInter10412_dataset.txt', 'C:\\Users\\yuhan\\Desktop\\data\\generated_data\\NPInter_10412\\')

    #get_RPI1807('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\', 'C:\\Users\\yuhan\\Desktop\\data\\generated_data\\RPI1807\\')
    # get_RPI369('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\RPI369_all.txt', 'C:\\Users\\yuhan\\Desktop\\data\\generated_data\\RPI369\\')

    # get_RPI13254('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\','C:\\Users\\yuhan\\Desktop\\data\\generated_data\\RPI13254\\')
    # get_RPI7317('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\RPI7317.csv','C:\\Users\\yuhan\\Desktop\\data\\generated_data\\RPI7317\\')

    # get_RPI1446('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\RPI1446_pairs.txt',
    #           'C:\\Users\\yuhan\\Desktop\\data\\generated_data\\RPI1446\\')

    # get_RPI2241('data\\raw_data\\','data\\generated_data\\RPI2241\\')

    #get_NPInter4158('C:\\Users\\yuhan\\Desktop\\data\\raw_data\\',
    #         'C:\\Users\\yuhan\\Desktop\\data\\generated_data\\NPInter_4158\\')
    get_RPI369('data\\raw_data\\RPI369_all.txt','data\\generated_data\\RPI369\\')



