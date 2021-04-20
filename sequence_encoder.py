from itertools import product

# encoder for protein sequence
class ProEncoder:
    elements = 'AIYHRDC'
    structs = 'hec'

    # clusters: {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
    #pro_intab = 'AGVILFPYMTSHNQWRKDEC'
    #pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

    def __init__(self, seq_k_upper_limit,struc_k_upper_limit):

        self.seq_k_upper_limit = seq_k_upper_limit #序列需要计算的最大的k值，算1-kmers
        self.struc_k_upper_limit = struc_k_upper_limit # 二级结构序列需要计算的最大的k值，算1-kmers
        self.seq_base = 7
        self.struct_base = 3

        self.seq_transDict = {}
        index = 0
        tar_list = ['0', '1', '2', '3', '4', '5', '6']
        groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
        for group in groups:
            g_members = sorted(group)
            for c in g_members:
                self.seq_transDict[c] = str(tar_list[index])
            index = index + 1

        self.struct_transDict = {'h':'0','e':'1','c':'2'}

        self.seq_all_kmers_map = self.get_all_kmers(self.seq_k_upper_limit,isStruc=False) #所有的1-kmers字典
        self.struc_all_kmers_map = self.get_all_kmers(self.struc_k_upper_limit,isStruc=True) #所有的1-kmers字典

    def get_all_kmers(self,k_upper_limit, isStruc=True):
        '''
        :param k_upper_limit:k的变化范围的上限
        :param isStruc:是否为二级结构
        :return:所有用数字表示的kmer字典
        '''
        kmers_map = {}
        if isStruc:
            base = '012'
        else:
            base = '0123456'
        for k in range(k_upper_limit):
            kmers_map[k + 1] = []
            for i in product(base, repeat=k + 1):
                kmers_map[k + 1].append(''.join(i))
        return kmers_map

    def translate_sequnce(self,sequence,TranslationDict):
        '''
        将序列翻译成数字表示的
        :param sequence: 翻译前的序列
        :param TranslationDict: 翻译依照的字典
        :return: 翻译后的序列
        '''
        from_list = []
        to_list = []
        for k, v in TranslationDict.items():
            from_list.append(k)
            to_list.append(v)
        trans_seq = sequence.translate(str.maketrans(str(from_list), str(to_list)))
        return trans_seq

    def calculate_kmer(self,sequence, base, all_kmers):
        '''
        计算单个的kmer
        :param sequence: 要计算kmer的序列
        :param base: 基的个数
        :param all_kmers: 所有的kmer列表
        :return:
        '''
        k = len(all_kmers[0])
        tmp_fea = [0] * (base**k)
        for x in range(len(sequence) + 1 - k):
            kmer = sequence[x:x + k]
            if kmer in all_kmers:
                index = all_kmers.index(kmer)
                tmp_fea[index] = tmp_fea[index] + 1
        feature = [float(val) / (len(sequence) + 1 - k) for val in tmp_fea]
        # 用所有的trids出现的次数进行规范化,也就是一条序列有多少个trids
        return feature

    def get_single_feature(self,sequence, isStruc, all_kmers):
        '''
        翻译序列并计算翻译后序列的kmer
        :param sequence:原始序列
        :param isStruc:是否是二级结构序列
        :param all_kmers:所有kmers
        :param k:
        :return: 得到某个kmer对应的特征
        '''
        if isStruc:
            TranslationDict = self.struct_transDict
            base = self.struct_base
        else:
            TranslationDict = self.seq_transDict
            base = self.seq_base
        trans_sequence = self.translate_sequnce(sequence, TranslationDict)

        feat = self.calculate_kmer(trans_sequence, base, all_kmers)
        return feat

    def get_kmer_feature(self,seq_sequence,struc_sequence):
        seq_feat = []
        struc_feat = []
        for k in range(self.seq_k_upper_limit):
            tmp = self.get_single_feature(seq_sequence, False, self.seq_all_kmers_map[k+1])
            seq_feat += tmp
        for k in range(self.struc_k_upper_limit):
            tmp = self.get_single_feature(struc_sequence,True, self.struc_all_kmers_map[k+1])
            struc_feat += tmp
        combined_feat = seq_feat+struc_feat

        return combined_feat, seq_feat, struc_feat


# encoder for RNA sequence
class RNAEncoder:
    elements = 'ACGU'
    structs = '.('

    def __init__(self, seq_k_upper_limit,struc_k_upper_limit):

        self.seq_k_upper_limit = seq_k_upper_limit #序列需要计算的最大的k值，算1-kmers
        self.struc_k_upper_limit = struc_k_upper_limit # 二级结构序列需要计算的最大的k值，算1-kmers
        self.seq_base = 4
        self.struct_base = 2

        self.seq_transDict = {'A':'0','C':'1','G':'2','U':'3'}
        self.struct_transDict = {'.':'0','(':'1'}

        self.seq_all_kmers_map = self.get_all_kmers(self.seq_k_upper_limit,isStruc=False) #所有的1-kmers字典
        self.struc_all_kmers_map = self.get_all_kmers(self.struc_k_upper_limit,isStruc=True) #所有的1-kmers字典

    def get_all_kmers(self,k_upper_limit, isStruc=True):
        '''
        :param k_upper_limit:k的变化范围的上限
        :param isStruc:是否为二级结构
        :return:所有用数字表示的kmer字典
        '''
        kmers_map = {}
        if isStruc:
            base = '01'
        else:
            base = '0123'
        for k in range(k_upper_limit):
            kmers_map[k + 1] = []
            for i in product(base, repeat=k + 1):
                kmers_map[k + 1].append(''.join(i))
        return kmers_map

    def translate_sequnce(self,sequence,TranslationDict,isStruc):
        '''
        将序列翻译成数字表示的
        :param sequence: 翻译前的序列
        :param TranslationDict: 翻译依照的字典
        :return: 翻译后的序列
        '''
        if isStruc:
            sequence.replace(')','(')

        from_list = []
        to_list = []

        for k, v in TranslationDict.items():
            from_list.append(k)
            to_list.append(v)
        trans_seq = sequence.translate(str.maketrans(str(from_list), str(to_list)))
        return trans_seq

    def calculate_kmer(self,sequence, base, all_kmers):
        '''
        计算单个的kmer
        :param sequence: 要计算kmer的序列
        :param base: 基的个数
        :param all_kmers: 所有的kmer列表
        :return:
        '''
        k = len(all_kmers[0])
        tmp_fea = [0] * (base**k)
        for x in range(len(sequence) + 1 - k):
            kmer = sequence[x:x + k]
            if kmer in all_kmers:
                index = all_kmers.index(kmer)
                tmp_fea[index] = tmp_fea[index] + 1
        feature = [float(val) / (len(sequence) + 1 - k) for val in tmp_fea]
        # 用所有的trids出现的次数进行规范化,也就是一条序列有多少个trids
        return feature

    def get_single_feature(self,sequence, isStruc, all_kmers):
        '''
        翻译序列并计算翻译后序列的kmer
        :param sequence:原始序列
        :param isStruc:是否是二级结构序列
        :param all_kmers:所有kmers
        :param k:
        :return: 得到某个kmer对应的特征
        '''
        if isStruc:
            TranslationDict = self.struct_transDict
            base = self.struct_base
        else:
            TranslationDict = self.seq_transDict
            base = self.seq_base
        trans_sequence = self.translate_sequnce(sequence, TranslationDict,isStruc)

        feat = self.calculate_kmer(trans_sequence, base, all_kmers)
        return feat

    def get_kmer_feature(self,seq_sequence,struc_sequence):
        seq_feat = []
        struc_feat = []
        for k in range(self.seq_k_upper_limit):
            tmp = self.get_single_feature(seq_sequence, False, self.seq_all_kmers_map[k + 1])
            seq_feat += tmp

        for k in range(self.struc_k_upper_limit):
            tmp = self.get_single_feature(struc_sequence, True, self.struc_all_kmers_map[k + 1])
            struc_feat += tmp

        combined_feat = seq_feat + struc_feat
        #print(combined_feat)
        return combined_feat, seq_feat, struc_feat



