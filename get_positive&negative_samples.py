import numpy as np
import pandas as pd
import math

def calculate_protein_sw_similarity(pr1, pr2, swscore_matrix):
    '''计算每对蛋白质之间规范化后的SmithWaterman similarity'''
    score = swscore_matrix[pr1,pr2]/math.sqrt(swscore_matrix[pr1,pr1]*swscore_matrix[pr2,pr2])
    return score

def get_positive_samples(NPI_filepath):
    NPInter = pd.read_table(NPI_filepath)
    protein = NPInter['UNIPROT-ID'].unique().tolist()  # 蛋白质列表
    ncRNA = NPInter['NONCODE-ID'].unique().tolist()  # RNA列表
    positive_index = []  # 相互作用对的下标[ncRNA_index,protein_index,label]
    for index, row in NPInter.iterrows():
        i = ncRNA.index(row['NONCODE-ID'])
        j = protein.index(row['UNIPROT-ID'])
        positive_index.append([i, j])
    return positive_index, protein, ncRNA

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

def get_Positives_and_Negatives (positive_samples, pr_list, RNA_list,swscore_matrix):
    Positives = []
    Negatives = []

    #random pair
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

    return Positives, Negatives

if __name__ == '__main__':
    NPInter_filepath = 'C:\\Users\\yuhan\\Desktop\\GNNAE\\raw_data\\NPInter10412_dataset.txt'
    swpath = 'C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\protein sw_smilarity matrix.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values
    positive_samples,protein, ncRNA = get_positive_samples(NPInter_filepath)
    '''
    Positives, Negatives = get_Positives_and_Negatives(positive_samples, protein, ncRNA, swscore_matrix)
    df1 = pd.DataFrame(Positives,columns = ['RNA','protein','label'])
    df2 = pd.DataFrame(Negatives,columns = ['RNA','protein','label'])
    inter = pd.merge(df1, df2, on=['RNA','protein'], how='inner')
    print(inter)
    df1.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\Positives.csv',index=False)
    df2.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\Negatives.csv',index=False)
    '''
    Positives = pd.read_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\Positives.csv')
    Negatives = pd.read_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\Negatives.csv')
    #Negatives = Negatives[:(len(Positives))]#按score升序选取和正样本一样多的负样本
    #Negatives = Negatives.sample(n=len(Positives),random_state=1) #随机抽取负样本
    #先按照分数取正样本两倍的负样本，再在其中抽取等量正样本
    Negatives = Negatives[:len(Positives)*2]
    Negatives = Negatives.sample(n=len(Positives), random_state=1)

    Negatives['label'] = -1
    edgelist = [Positives,Negatives]
    edgelist = pd.concat(edgelist,axis=0)
    edgelist = edgelist.reset_index(drop=True)
    np.savetxt('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\graph.edgelist.txt', edgelist, fmt='%s', delimiter=' ')
    edgelist.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\edgelist.csv',header=None)
    NPI_pos, NPI_neg = np.zeros((len(ncRNA),len(protein))), np.zeros((len(ncRNA),len(protein)))
    NPI_pos[Positives.values[:,0],Positives.values[:,1]] = 1
    NPI_neg[Negatives.values[:,0],Negatives.values[:,1]] = 1
    NPI_pos = pd.DataFrame(NPI_pos)
    NPI_neg = pd.DataFrame(NPI_neg)
    NPI_pos.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\NPI_pos.csv',index=False, header=None)
    #NPI_neg.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\NPI_neg.csv', index=False, header=None)
    #NPI_neg.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\NPI_neg_random.csv', index=False, header=None)
    NPI_neg.to_csv('C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\NPI_neg_selected_random.csv', index=False, header=None)







