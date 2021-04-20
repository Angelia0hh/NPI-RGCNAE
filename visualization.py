import pandas as pd
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

identity_feature_dim = 1024
filepath = 'data/generated_data/NPInter_10412/'
NPI_pos_matrix = pd.read_csv(filepath + 'NPI_pos.csv', header=None).values

onehot = np.identity(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], dtype=np.float32)
RNA_onehot, protein_onehot = onehot[:NPI_pos_matrix.shape[0]], \
                             onehot[NPI_pos_matrix.shape[0]:]

random = np.random.randn(NPI_pos_matrix.shape[0] + NPI_pos_matrix.shape[1], identity_feature_dim)
RNA_random, protein_random = random[:NPI_pos_matrix.shape[0]], \
                             random[NPI_pos_matrix.shape[0]:]
tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)

RNA_tsne_onehot = tsne.fit_transform(RNA_onehot)
protein_tsne_onehot = tsne.fit_transform(protein_onehot)
print("Org data dimension is {}.Embedded data dimension is {}".format(RNA_onehot.shape[-1], RNA_tsne_onehot.shape[-1]))
protein_tsne_onehot = pd.DataFrame(protein_tsne_onehot)
RNA_tsne_onehot = pd.DataFrame(RNA_tsne_onehot)
protein_tsne_onehot.to_csv('result/NPInter_10412/coordinates_protein_onehot.csv',index=False)
RNA_tsne_onehot.to_csv('result/NPInter_10412/coordinates_RNA_onehot.csv',index=False)

RNA_tsne_random = tsne.fit_transform(RNA_random)
protein_tsne_random = tsne.fit_transform(protein_random)
print("Org data dimension is {}.Embedded data dimension is {}".format(RNA_random.shape[-1], RNA_tsne_random.shape[-1]))
protein_tsne_random = pd.DataFrame(protein_tsne_random)
RNA_tsne_random = pd.DataFrame(RNA_tsne_random)
protein_tsne_random.to_csv('result/NPInter_10412/coordinates_protein_random.csv',index=False)
RNA_tsne_random.to_csv('result/NPInter_10412/coordinates_RNA_random.csv',index=False)

RNA_tsne_onehot = pd.read_csv('result/NPInter_10412/coordinates_RNA_onehot.csv',header=None).values
protein_tsne_onehot = pd.read_csv('result/NPInter_10412/coordinates_protein_onehot.csv',header=None).values

RNA_tsne_random = pd.read_csv('result/NPInter_10412/coordinates_RNA_random.csv',header=None).values
protein_tsne_random = pd.read_csv('result/NPInter_10412/coordinates_protein_random.csv',header=None).values


def visualized3D(RNA_tsne,protein_tsne):

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

visualized3D(RNA_tsne_onehot,protein_tsne_onehot)
visualized3D(RNA_tsne_random,protein_tsne_random)