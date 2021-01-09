import pandas as pd
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import time
'''
#tsne 降维
tsne = manifold.TSNE(n_components=3, init='pca', random_state=1)
RNA = pd.read_csv('result/NPInter_10412/embeddings_RNA.csv',header=None)
protein = pd.read_csv('result/NPInter_10412/embeddings_protein.csv',header=None)
RNA_tsne = tsne.fit_transform(RNA)
protein_tsne = tsne.fit_transform(protein)
print("Org data dimension is {}.Embedded data dimension is {}".format(RNA.shape[-1], RNA_tsne.shape[-1]))
protein_tsne = pd.DataFrame(protein_tsne)
RNA_tsne = pd.DataFrame(RNA_tsne)
protein_tsne.to_csv('result/NPInter_10412/coordinates_protein.csv',index=False)
RNA_tsne.to_csv('result/NPInter_10412/coordinates_RNA.csv',index=False)
'''


RNA_tsne = pd.read_csv('result/NPInter_10412/coordinates_RNA.csv',header=None).values
protein_tsne = pd.read_csv('result/NPInter_10412/coordinates_protein.csv',header=None).values

#三维可视化
fig = plt.figure()
ax = Axes3D(fig)
# 归一化
RNA_min, RNA_max = RNA_tsne.min(0), RNA_tsne.max(0)
RNA_norm = (RNA_tsne - RNA_min) / (RNA_max - RNA_min)
protein_min, protein_max = protein_tsne.min(0), protein_tsne.max(0)
protein_norm = (protein_tsne - protein_min) / (protein_max - protein_min)
#画总的三维图
#ax.scatter3D(protein_tsne[:,2], protein_tsne[:,0], protein_tsne[:,1], cmap='Greens')
#ax.scatter3D(RNA_tsne[:,2], RNA_tsne[:,0], RNA_tsne[:,1], cmap='Greens')
ax.scatter3D(protein_tsne[:,0], protein_tsne[:,1], protein_tsne[:,2], cmap='Greens',label='protein')
ax.scatter3D(RNA_tsne[:,0], RNA_tsne[:,1], RNA_tsne[:,2], cmap='Greens',label='RNA')
ax.set_xlabel('X Label',fontsize=11)
ax.set_ylabel('Y Label',fontsize=11)
ax.set_zlabel('Z Label',fontsize=11)
ax.xaxis.set_major_locator(MultipleLocator(160))
ax.yaxis.set_major_locator(MultipleLocator(160))
ax.zaxis.set_major_locator(MultipleLocator(300))
ax.set_title('RNA and Protein Node Embeddings',fontsize=11)
ax.legend(loc='lower left',fontsize=11)
plt.show()

'''
#二维
RNA_min, RNA_max = RNA_tsne.min(0), RNA_tsne.max(0)
RNA_norm = (RNA_tsne - RNA_min) / (RNA_max - RNA_min)
protein_min, protein_max = protein_tsne.min(0), protein_tsne.max(0)
protein_norm = (protein_tsne - protein_min) / (protein_max - protein_min)
plt.scatter(protein_norm[:,0], protein_norm[:,1], marker='o', label="circle")
plt.scatter(RNA_norm[:,0], RNA_norm[:,1], marker='^', label="triangle")
plt.legend(loc='best')
plt.show()
'''


# 归一化
RNA_tsne = pd.DataFrame(RNA_tsne)
protein_tsne = pd.DataFrame(protein_tsne)

protein_part1 = protein_tsne[protein_tsne[0]>0]
protein_part2 = protein_tsne[protein_tsne[0]<0]
protein_part3 = protein_tsne[protein_tsne[0]==0]
print("截取后的样本:{},{},{}".format(len(protein_part1),len(protein_part2),len(protein_part3)))
RNA_part1 = RNA_tsne[RNA_tsne[0]>0]
RNA_part2 = RNA_tsne[RNA_tsne[0]<0]
RNA_part3 = RNA_tsne[RNA_tsne[0]==0]
print("截取后的样本:{},{},{}".format(len(RNA_part1),len(RNA_part2),len(RNA_part3)))


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(protein_part1[0], protein_part1[1], protein_part1[2], cmap='Greens',label='protein')
ax.scatter3D(RNA_part1[0], RNA_part1[1], RNA_part1[2], cmap='Greens',label='RNA')
ax.set_xlabel('X Label',fontsize=11)
ax.set_ylabel('Y Label',fontsize=11)
ax.set_zlabel('Z Label',fontsize=11)
ax.xaxis.set_major_locator(MultipleLocator(160))
ax.yaxis.set_major_locator(MultipleLocator(160))
ax.zaxis.set_major_locator(MultipleLocator(300))
ax.set_title('Part of RNA and Protein Node Embeddings',fontsize=11)
ax.legend(loc='lower left',fontsize=11)
plt.show()

plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(protein_part2[0], protein_part2[1], protein_part2[2], cmap='Greens',label='protein')
ax.scatter3D(RNA_part2[0], RNA_part2[1], RNA_part2[2], cmap='Greens',label='RNA')
ax.set_xlabel('X Label',fontsize=11)
ax.set_ylabel('Y Label',fontsize=11)
ax.set_zlabel('Z Label',fontsize=11)
ax.xaxis.set_major_locator(MultipleLocator(160))
ax.yaxis.set_major_locator(MultipleLocator(160))
ax.zaxis.set_major_locator(MultipleLocator(300))
ax.set_title('Part of RNA and Protein Node Embeddings',fontsize=11)
ax.legend(loc='lower left',fontsize=11)
plt.show()

plt.show()

