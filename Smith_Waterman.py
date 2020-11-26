import re
import os
import numpy as np
import pandas as pd
import multiprocessing

match = 3
mismatch = -3
gap = -2 #空位罚分 空位权值恒定模型

# 蛋白质替换记分矩阵用BLOSUM-62
S_matrix = [[9,-1,-1,-3,0,-3,-3,-3,-4,-3,-3,-3,-3,-1,-1,-1,-1,-2,-2,-2],
     [-1,4,1,-1,1,0,1,0,0,0,-1,-1,0,-1,-2,-2,-2,-2,-2,-3],
     [-1,1,4,1,-1,1,0,1,0,0,0,-1,0,-1,-2,-2,-2,-2,-2,-3],
     [-3,-1,1,7,-1,-2,-1,-1,-1,-1,-2,-2,-1,-2,-3,-3,-2,-4,-3,-4],
     [0,1,-1,-1,4,0,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-2,-3],
     [-3,0,1,-2,0,6,-2,-1,-2,-2,-2,-2,-2,-3,-4,-4,0,-3,-3,-2],
     [-3,1,0,-2,-2,0,6,1,0,0,-1,0,0,-2,-3,-3,-3,-3,-2,-4],
     [-3,0,1,-1,-2,-1,1,6,2,0,-1,-2,-1,-3,-3,-4,-3,-3,-3,-4],
     [-4,0,0,-1,-1,-2,0,2,5,2,0,0,1,-2,-3,-3,-3,-3,-2,-3],
     [-3,0,0,-1,-1,-2,0,0,2,5,0,1,1,0,-3,-2,-2,-3,-1,-2],
     [-3,-1,0,-2,-2,-2,1,1,0,0,8,0,-1,-2,-3,-3,-2,-1,2,-2],
     [-3,-1,-1,-2,-1,-2,0,-2,0,1,0,5,2,-1,-3,-2,-3,-3,-2,-3],
     [-3,0,0,-1,-1,-2,0,-1,1,1,-1,2,5,-1,-3,-2,-3,-3,-2,-3],
     [-1,-1,-1,-2,-1,-3,-2,-3,-2,0,-2,-1,-1,5,1,2,-2,0,-1,-1],
     [-1,-2,-2,-3,-1,-4,-3,-3,-3,-3,-3,-3,-3,1,4,2,1,0,-1,-3],
     [-1,-2,-2,-3,-1,-4,-3,-4,-3,-2,-3,-2,-2,2,2,4,3,0,-1,-2],
     [-1,-2,-2,-2,0,-3,-3,-3,-2,-2,-3,-3,-2,1,3,1,4,-1,-1,-3],
     [-2,-2,-2,-4,-2,-3,-3,-3,-3,-3,-1,-3,-3,0,0,0,-1,6,3,1],
     [-2,-2,-2,-3,-2,-3,-2,-3,-2,-1,2,-2,-2,-1,-1,-1,-1,3,7,2],
     [-2,-3,-3,-4,-3,-2,-4,-4,-3,-2,-2,-3,-3,-1,-3,-2,-3,1,2,11]]

amino_acid = ['C','S','T','P','A', 'G', 'N','D','E','Q','H','R','K',
            'M','I','L','V','F','Y','W']





def read_fasta_file(fasta_file):
    '''
    读取protein sequence的fasta格式数据
    :param fasta_file: fasta数据所在地址
    :return: key为蛋白质名称，value为sequence的字典
    '''
    fp = open(fasta_file, 'r')
    allsequences = []
    sequence = ''
    i = 0
    for line in fp:
        # let's discard the newline at the end (if any)
        line = line.rstrip().strip('*')
        if line == '':
            pass
        # distinguish header from sequence
        elif line[0] == '>':  # or line.startswith('>')
            if i == 0:
                i += 1
            else:
                allsequences.append(sequence)
                sequence = ''
        else:
            # it is sequence
            sequence += line
    allsequences.append(sequence)
    fp.close()
    print(len(allsequences))
    return allsequences


def s_w(seqA, allseq, savepath, num):
    #num 序列的index
    scorelist = [0]*(num) # seqA之前的序列已经比较过，得分直接置0
    print('Comparing the %d sequence'%(num+1))
    # 计算得分矩阵
    cols = len(seqA)
    for seqB in allseq:
        rows = len(seqB)
        matrix = [[0 for row in range(rows+1)] for col in range(cols+1)]
        paths = [[0 for row in range(rows+1)] for col in range(cols+1)]
        max_score = 0
        finalscore = 0
        for i in range(cols):
            for j in range(rows):
                a1 = amino_acid.index(seqA[i])
                a2 = amino_acid.index(seqB[j])
                s = S_matrix[a1][a2]
                if seqA[i] == seqB[j]:
                    diag = matrix[i][j] + s
                else:
                    diag = matrix[i][j] + s
                up = matrix[i + 1][j] + gap
                left = matrix[i][j + 1] + gap
                score = max(0,diag, up, left)
                matrix[i+1][j+1] = score
                if score > max_score:
                    max_score = score
                    start_pos = [i+1, j+1]
                if matrix[i+1][j+1] == diag and matrix[i+1][j+1] != 0:
                    paths[i+1][j+1] = 'diag'
                elif matrix[i+1][j+1] == up   and matrix[i+1][j+1] != 0:
                    paths[i+1][j+1] = 'up'
                elif matrix[i+1][j+1] == left and matrix[i+1][j+1] != 0:
                    paths[i+1][j+1] = 'left'
        #根据path回溯计算得分
        i, j = start_pos
        start_path = paths[i][j]
        while start_path != 0:
            finalscore += matrix[i][j]
            if start_path == 'diag':
                i, j = i-1, j-1
            elif start_path == 'up':
                j = j-1
            else:
                i = i-1
            start_path = paths[i][j]
        scorelist.append(finalscore)
    np.savetxt(savepath, scorelist, delimiter=',', fmt='%f')


def generated_SW_matrix(filename,path):
    allsequence = read_fasta_file(filename)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for i in range(len(allsequence)):
        savepath = path + str(i + 1) + '.txt'
        sequence1 = allsequence[i]
        sequence2 = allsequence[i:]
        pool.apply_async(s_w, (sequence1, sequence2, savepath, i,))
    pool.close()
    pool.join()

    scorematrix = []
    for i in range(len(allsequence)):
        alignpath = path + str(i + 1) + '.txt'
        alignlist = pd.read_csv(alignpath, header=None, index_col=None)
        alignlist = np.array(alignlist)
        alignlist = alignlist.T
        scorematrix.append(alignlist[0])
    finalmatrix = np.array(scorematrix)
    for j in range(finalmatrix.shape[1]):
        for i in range(finalmatrix.shape[0]):
            finalmatrix[i][j] = finalmatrix[j][i]
    np.savetxt(os.path.join(path, r'protein sw_smilarity matrix.csv'), finalmatrix, delimiter=',', fmt='%f')
    # np.savetxt(os.path.join(path, r'protein sw_test.csv'), finalmatrix, delimiter=',', fmt='%f')

if __name__ == '__main__':
    generated_SW_matrix(filename = 'C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\NPInter_4158\\protein_extracted_seq.fasta',
                        path = 'C:\\Users\\yuhan\\Desktop\\GNNAE\\generated_data\\NPInter_4158\\')
