import pandas as pd
import numpy as np
#读入每对节点
#计算边两端的节点相连的一阶邻居数量
path = 'data/generated_data/'
def read_data(filepath,dataset,isPos,negative_method):
    edgelist = pd.read_csv(filepath + dataset + "/edgelist_" + negative_method + ".csv", header=None,
                           names=['index', 'RNA', 'protein', 'label'])
    if isPos:
        adj = pd.read_csv(filepath+dataset+'/NPI_pos.csv',header=None).values
        pair_list = edgelist[edgelist['label'] == 1][['RNA', 'protein']]
    else:
        adj = pd.read_csv(filepath + dataset + '/NPI_neg_'+negative_method+'.csv', header=None).values
        pair_list = edgelist[edgelist['label'] == -1][['RNA', 'protein']]
    return adj,pair_list

def cal_neighbor_num(pair_list,adj):
    '''

    :param pair_list: 每对连接的名字，RNA在前，蛋白质在后
    :param adj:
    :return:
    '''
    neighbors = 0
    RNA_deg = adj.sum(1)
    protein_deg = adj.sum(0)
    for ind,row in pair_list.iterrows():
        neighbors += RNA_deg[row['RNA']]
        neighbors += protein_deg[row['protein']]
        
    return neighbors
    
def run(filepath, dataset, isPos, negative_method):
    adj_pos,pair_list_pos = read_data(filepath, dataset, True, negative_method)
    neighbors_pos = cal_neighbor_num(pair_list_pos, adj_pos)
    adj_neg, pair_list_neg = read_data(filepath, dataset, False, negative_method)
    neighbors_neg = cal_neighbor_num(pair_list_neg, adj_neg)
    avg_neighbors = (neighbors_pos+neighbors_neg)/(len(pair_list_pos)*2)
    print(avg_neighbors)
    
    return avg_neighbors


print("RPI2241:")

print("random:")
run(path, 'RPI2241', isPos=False, negative_method='random')
print("sort random:")
run(path, 'RPI2241', isPos=False, negative_method='sort_random')
print("sort:")
run(path, 'RPI2241', isPos=False, negative_method='sort')
print("-------------------------------------")

print("RPI369")

print("random:")
run(path, 'RPI369', isPos=False, negative_method='random')
print("sort random:")
run(path, 'RPI369', isPos=False, negative_method='sort_random')
print("sort:")
run(path, 'RPI369', isPos=False, negative_method='sort')
print("-------------------------------------")

print("NPInter10412")
print("random:")
run(path, 'NPInter_10412', isPos=False, negative_method='random')
print("sort random:")
run(path, 'NPInter_10412', isPos=False, negative_method='sort_random')
print("sort:")
run(path, 'NPInter_10412', isPos=False, negative_method='sort')
print("-------------------------------------")

print("RPI7317")
print("random:")
run(path, 'RPI7317', isPos=False, negative_method='random')
print("sort random:")
run(path, 'RPI7317', isPos=False, negative_method='sort_random')
print("sort:")
run(path, 'RPI7317', isPos=False, negative_method='sort')


