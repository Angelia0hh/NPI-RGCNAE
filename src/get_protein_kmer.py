import numpy as np
import pandas as pd
def read_fasta_file(fasta_file):
    '''
    读取protein sequence的fasta格式数据
    :param fasta_file: fasta数据所在地址
    :return: key为蛋白质名称，value为sequence的字典
    '''
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    # pdb.set_trace()
    for line in fp:
        # let's discard the newline at the end (if any)
        line = line.rstrip().strip('*')
        if line == '':
            pass
        # distinguish header from sequence
        elif line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:]  # discarding the initial >
            seq_dict[name] = ''
        else:
            # it is sequence
            seq_dict[name] = seq_dict[name] + line
    fp.close()

    print(len(seq_dict))
    '''
    for key, value in seq_dict.items():
        print(key)
        print(value)
    '''
    return seq_dict

def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    seq_len = len(seq)
    tri_feature = []

    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [float(val) / (len(seq) + 1 - k) for val in tmp_fea] #用所有的trids出现的次数进行规范化,也就是一条序列有多少个trids
        # pdb.set_trace()
    return tri_feature

def TransDict_from_list(groups):
    '''
    得到字母和7个数字对应的字典
    '''
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    # groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    for group in groups:
        g_members = sorted(group)
        #print(g_members)
        for c in g_members:
            result[c] = str(tar_list[index])
        index = index + 1

    return result

def translate_sequence (seq, TranslationDict):
    '''
    将序列转换为由7个数字表示的新序列
    '''
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))

    return TRANS_seq

def find_all_path(path,cnt,base,all_kmers,k):
    if(cnt>k):
        all_kmers.append(path)
    else:
        for i in range(base):
            path+=str(i)
            find_all_path(path,cnt+1,base,all_kmers,k)
            path = path[:-1]

def get_k_protein_trids(k):
    '''
    得到所有的protein的kmer组合，用7个数字表示
    '''
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    all_kmers = []
    cnt = 1
    for j in range(base):
        path = ""
        path += str(j)
        find_all_path(path, cnt + 1, base, all_kmers,k)

    return all_kmers

def generated_protein_kmer(fasta_file, savepath, k = 3):
    Protein3mer = []
    protein_seq_dict = read_fasta_file(fasta_file)
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_k_protein_trids(3)

    for protein, protein_seq in protein_seq_dict.items():
        protein_seq1 = translate_sequence(protein_seq, group_dict)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq1, pythoncount=False)
        Protein3mer.append(protein_tri_fea)
    Protein3mer = np.array(Protein3mer)
    print(Protein3mer.shape)
    Protein3mer = pd.DataFrame(Protein3mer)
    #Protein3mer.to_csv(savepath, index=False)

if __name__ == '__main__':

    generated_protein_kmer(
        fasta_file='../data/generated_data/RPI369/protein_extracted_seq.fasta',
        savepath='../data/generated_data/RPI369/Protein3merfeat.csv', k=3)
    generated_protein_kmer(
        fasta_file='../data/generated_data/RPI2241/protein_extracted_seq.fasta',
        savepath='../data/generated_data/RPI2241/Protein3merfeat.csv', k=3)
    generated_protein_kmer(
        fasta_file='../data/generated_data/RPI7317/protein_extracted_seq.fasta',
        savepath='../data/generated_data/RPI7317/Protein3merfeat.csv', k=3)
    generated_protein_kmer(
        fasta_file='../data/generated_data/NPInter_10412/protein_extracted_seq.fasta',
        savepath='../data/generated_data/NPInter_10412/Protein3merfeat.csv', k=3)
    generated_protein_kmer(
        fasta_file='../data/generated_data/NPInter_4158/protein_extracted_seq.fasta',
        savepath='../data/generated_data/NPInter_4158/Protein3merfeat.csv', k=3)




