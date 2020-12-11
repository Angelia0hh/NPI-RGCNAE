import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
from pandas import DataFrame


res_path = 'data/generated_data/NPInter_10412/result/predictions/'
path_list = []
for root,dir,files in os.walk(res_path):
    files = files
print(files)
for file in files:
    path_list.append(res_path+file)

mps = {}
for i,res_file in enumerate(path_list):  # 利用模型划分数据集和目标变量 为一一对应的下标
    res = pd.read_csv(res_file)
    cnt = 0
    mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 100)
    for j in range(5):
        cnt += 1
        fpr, tpr, thresholds = roc_curve(res['label '+str(j+1)], res['pred '+str(j+1)])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
        plt.figure(i)
        roc_auc = auc(fpr, tpr)  # 求auc面积
        plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.4f})'.format(j+1, roc_auc))  # 画出当前分割数据的ROC曲线

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线

    mean_tpr /= cnt  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.4f})'.format(mean_auc), lw=2)

    mps[files[i][5:-4]] = (mean_fpr,mean_tpr)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('{}'.format(files[i][5:-4]))
    plt.legend(loc="lower right")
    plt.show()

# 所有的方法汇总画图
plt.figure(figsize=(12,8), dpi=600,facecolor=(1, 1, 1))
makes = itertools.cycle(['.',',','o','v','^','<', '>','1','2','3','4','s','p','*','h','H','+','x','D','d' ,'|' ,'_'])
colors = ['']
plt.figure(25)
names = ['sort_random_one hot_stack_side','sort_random_one hot_sum_side','sort_random_one hot_sum','sort_random_one hot_stack',
         'random_node2vec_stack', 'random_node2vec_stack_side', 'random_node2vec_sum', 'random_node2vec_sum_side',
         'sort_node2vec_stack', 'sort_node2vec_stack_side', 'sort_node2vec_sum', 'sort_node2vec_sum_side',
        'sort_random_node2vec_stack_side', 'sort_random_node2vec_sum', 'sort_random_node2vec_sum_side', 'sort_random_node2vec_stack'
         ]
colors = ['']
for label,data in mps.items():
    fpr = data[0]
    tpr = data[1]
    if label in names:
        mean_auc = auc(fpr, tpr)
        if label.endswith("stack"):
            plt.plot(fpr, tpr,marker = '*',markersize=2,label=label +' (area = {0:.4f})'.format(mean_auc), lw=1)
        elif label.endswith('stack_side'):
            plt.plot(fpr, tpr, marker = 'p',markersize=2, label=label + ' (area = {0:.4f})'.format(mean_auc), lw=1)
        elif label.endswith('sum'):
            plt.plot(fpr, tpr, marker ='o', markersize=2,label=label + ' (area = {0:.4f})'.format(mean_auc), lw=1)
        elif label.endswith('sum_side'):
            plt.plot(fpr, tpr, marker = 'v', markersize=2,label=label + ' (area = {0:.4f})'.format(mean_auc), lw=1)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC curve of different model combinations')
plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)

plt.savefig('data/generated_data/NPInter_10412/result/pictures/ROC curve.svg',bbox_inches='tight')
plt.savefig('data/generated_data/NPInter_10412/result/pictures/ROC curve.png',bbox_inches='tight')

plt.show()