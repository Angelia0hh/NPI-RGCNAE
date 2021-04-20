import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
from pandas import DataFrame


plt.rc('font',family='Times New Roman')
res_path = '../results/NPInter_10412/combination/'
def calculate_coordinate(res_path = '../results/NPInter_10412/combination/'):
    path_list = []
    files = []
    for file in os.listdir(res_path):
        if file.startswith("prob") and file.__contains__("random"):
            print(file)
            files.append(file)
    for file in files:
        path_list.append(res_path+file)

    mps = {}
    for i, res_file in enumerate(path_list):  # 利用模型划分数据集和目标变量 为一一对应的下标
        res = pd.read_csv(res_file)

        fpr, tpr, thresholds = roc_curve(res['label'], res['pred'])
        mps[files[i][5:-4]] = (fpr, tpr)

    return mps

mps = calculate_coordinate('../results/NPInter_10412/combination/')
mps2 =  calculate_coordinate('../results/RPI369/combination/')
# 所有的方法汇总画图
plt.figure(figsize=(18,5), dpi=300,facecolor=(1, 1, 1))
plt.tight_layout()
plt.subplots_adjust(wspace =0.2, hspace =0.2)
plt.subplot(121)

names = ['sort_random_stack_side', 'sort_random_stack_withoutside']

for label,data in mps.items():

    fpr = data[0]
    tpr = data[1]
    if label in names:

        comb = label.split('_')
        if label.endswith("withoutside"):
            plt.plot(fpr, tpr,label= "without side information" +' (area = {0:.4f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('side'):
            plt.plot(fpr, tpr,label= "with side information" + ' (area = {0:.4f})'.format(auc(fpr, tpr)), lw=1)
        '''
        elif label.endswith('sum_withoutside'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.4f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('sum_side'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.4f})'.format(auc(fpr, tpr)), lw=1)
        '''


plt.plot([0, 1], [0, 1], 'k--', label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=11)
plt.ylabel('True Positive Rate',fontsize=11)  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC curve of different feature combinations on NPInter 10412',fontsize=11)
#plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.legend(loc="lower right",fontsize=7)


plt.subplot(122)
for label, data in mps2.items():

    fpr = data[0]
    tpr = data[1]
    if label in names:
        comb = label.split('_')
        if label.endswith("withoutside"):
            plt.plot(fpr, tpr, label="without side information" + ' (area = {0:.4f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('side'):
            plt.plot(fpr, tpr, label="with side information"+ ' (area = {0:.4f})'.format(auc(fpr, tpr)), lw=1)

plt.plot([0, 1], [0, 1], 'k--', label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC curve of different feature combinations on RPI369', fontsize=11)
# plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.legend(loc="lower right", fontsize=7)

plt.savefig('figures/ROC curve of different combinations.svg',bbox_inches='tight')
plt.savefig('figures/ROC curve of different combinations.png',bbox_inches='tight')

plt.show()