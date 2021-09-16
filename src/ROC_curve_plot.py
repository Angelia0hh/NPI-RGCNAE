
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd



plt.rc('font',family='Times New Roman')
res_path = '..\\results\\NPInter_10412\\combination\\'
#res_path = 'D:\\PycharmProject\\NPI-RGCNAE\\result_8.30\\NPInter_10412\\combination\\'
def calculate_coordinate(res_path = 'results\\NPInter_10412\\combination\\'):
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
        '''
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        for j in range(5):
            fpr, tpr, thresholds = roc_curve(res['label '+str(j+1)], res['pred '+str(j+1)])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)# 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
            interp_tpr[0] = 0.0# 将第一个真正例=0 以0为起点
            tprs.append(interp_tpr)
            # plt.figure(i)
            # roc_auc = auc(fpr, tpr)  # 求auc面积
            # plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.4f})'.format(j+1, roc_auc))  # 画出当前分割数据的ROC曲线

        # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线

        mean_tpr = np.mean(tprs, axis=0) # 求数组的平均值
        mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
        '''
        fpr, tpr, thresholds = roc_curve(res['label'], res['pred'])
        mps[files[i][5:-4]] = (fpr, tpr)
        '''
        plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.4f})'.format(mean_auc), lw=2)
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        plt.title('{}'.format(files[i][5:-4]))
        plt.legend(loc="lower right")
        plt.show()
        '''
    #print(mps)
    return mps
mps1 = calculate_coordinate('..\\results\\RPI2241\\combination\\')
mps2 =  calculate_coordinate('..\\results\\RPI7317\\combination\\')
mps3 = calculate_coordinate('..\\results\\NPInter_10412\\combination\\')
mps4 = calculate_coordinate('..\\results\\RPI369\\combination\\')
'''
mps1 = calculate_coordinate('D:\\PycharmProject\\NPI-RGCNAE\\result_8.30\\RPI2241\\combination\\')
mps2 =  calculate_coordinate('D:\\PycharmProject\\NPI-RGCNAE\\result_8.30\\RPI7317\\combination\\')
mps3 = calculate_coordinate('D:\\PycharmProject\\NPI-RGCNAE\\result_8.30\\NPInter_10412\\combination\\')
mps4 = calculate_coordinate('D:\\PycharmProject\\NPI-RGCNAE\\result_8.30\\RPI369\\combination\\')
'''
# 所有的方法汇总画图
plt.figure(figsize=(16,12), dpi=600,facecolor=(1, 1, 1))
plt.tight_layout()
plt.subplots_adjust(wspace =0.2, hspace =0.2)
plt.subplot(221)

names = ['sort_random_stack_side', 'sort_random_stack_withoutside']

for label,data in mps1.items():

    fpr = data[0]
    tpr = data[1]
    if label in names:

        comb = label.split('_')
        if label.endswith("withoutside"):
            plt.plot(fpr, tpr,label= "without sequence-based features\n" +' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('side'):
            plt.plot(fpr, tpr,label= "with sequence-based features\n" + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        '''
        elif label.endswith('sum_withoutside'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('sum_side'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        '''


plt.plot([0, 1], [0, 1], 'k--', label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.tick_params(labelsize=14)
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)  # 可以使用中文，但需要导入一些库即字体
plt.title('(a)',fontsize=14,loc='center',y=-0.2)
#plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.legend(loc="lower right",fontsize=15)


plt.subplot(222)
for label, data in mps2.items():

    fpr = data[0]
    tpr = data[1]
    if label in names:
        comb = label.split('_')
        if label.endswith("withoutside"):
            plt.plot(fpr, tpr, label="without sequence-based features\n" + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('side'):
            plt.plot(fpr, tpr, label="with sequence-based features\n"+ ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)

plt.plot([0, 1], [0, 1], 'k--', label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.tick_params(labelsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('(b)',fontsize=14,loc='center',y=-0.2)
# plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.legend(loc="lower right", fontsize=15)

plt.subplot(223)
for label,data in mps3.items():

    fpr = data[0]
    tpr = data[1]
    if label in names:

        comb = label.split('_')
        if label.endswith("withoutside"):
            plt.plot(fpr, tpr,label= "without sequence-based features\n" +' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('side'):
            plt.plot(fpr, tpr,label= "with sequence-based features\n" + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        '''
        elif label.endswith('sum_withoutside'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('sum_side'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        '''


plt.plot([0, 1], [0, 1], 'k--', label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.tick_params(labelsize=14)
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)  # 可以使用中文，但需要导入一些库即字体
plt.title('(c)',fontsize=14,loc='center',y=-0.2)
#plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.legend(loc="lower right",fontsize=15)


plt.subplot(224)
for label,data in mps4.items():

    fpr = data[0]
    tpr = data[1]
    if label in names:

        comb = label.split('_')
        if label.endswith("withoutside"):
            plt.plot(fpr, tpr,label= "without sequence-based features\n" +' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('side'):
            plt.plot(fpr, tpr,label= "with sequence-based features\n" + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        '''
        elif label.endswith('sum_withoutside'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        elif label.endswith('sum_side'):
            plt.plot(fpr, tpr,label=comb[1]+"+"+comb[3] + ' (area = {0:.3f})'.format(auc(fpr, tpr)), lw=1)
        '''


plt.plot([0, 1], [0, 1], 'k--', label='Luck')  # 画对角线

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.tick_params(labelsize=14)
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)  # 可以使用中文，但需要导入一些库即字体
plt.title('(d)',fontsize=14,loc='center',y=-0.2)
#plt.legend(fontsize=7,bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.legend(loc="lower right",fontsize=15)

plt.savefig('results\\Fig3.svg',bbox_inches='tight')
plt.show()
