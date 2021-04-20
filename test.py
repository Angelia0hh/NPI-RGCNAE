# 为RPITER 7317生成fasta数据
def read_fasta_file(fasta_file):
    '''
    #读取protein sequence的fasta格式数据
    #:param fasta_file: fasta数据所在地址
    #:return: key为蛋白质名称，value为sequence的字典
    '''
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line == '':
            pass
        elif line[0] == '>':
            name = line[1:].split("|")[1]
            seq_dict[name] = ''
        else:
            # it is sequence
            seq_dict[name] = seq_dict[name] + line
    fp.close()
    '''
    #print(len(seq_dict))
    #for key, value in seq_dict.items():
    #    print(key)
    #    print(value)
    '''
    return seq_dict

'''
seq_dict = read_fasta_file('data/generated_data/RPI7317/protein_extracted_seq.fasta')
with open('data/generated_data/RPI7317/RPITER_RPI7317_protein_seq.fasta','w') as f:
    for name, seq in seq_dict.items():
        f.write(">"+name+"\n")
        f.write(seq+"\n")


struct_dict  = read_fasta_file('data/generated_data/RPI7317/protein_structure_seq.fasta')
with open('data/generated_data/RPI7317/RPITER_RPI7317_protein_structure.fasta','w') as f:
    for name, struct in struct_dict.items():
        f.write(">"+name+"\n")
        f.write(struct+"\n")


RNA_struct_dict = {}
with open('data/generated_data/RPI7317/ncRNA_structure_seq.fasta') as f:
    for line in f:
        if line.startswith(">"):
            name = line.rstrip()[1:]
            RNA_struct_dict[name] = ""
        elif line.startswith('(') or line.startswith(".") or line.startswith(")"):
            RNA_struct_dict[name] = RNA_struct_dict[name] + line.rstrip()
        else:
            pass

with open('data/generated_data/RPI7317/RPITER_RPI7317_RNA_structure.fasta','w') as f:
    for name, struct in RNA_struct_dict.items():
        f.write(">"+name+"\n")
        f.write(struct+"\n")
'''



RNA_struct_dict = {}
with open('data/generated_data/RPI7317/ncRNA_structure_seq.fasta') as f:
    for line in f:
        if line.startswith(">"):
            name = line.rstrip()[1:]
            RNA_struct_dict[name] = ""
        elif line.startswith('(') or line.startswith(".") or line.startswith(")"):
            RNA_struct_dict[name] = RNA_struct_dict[name] + line.rstrip()
        else:
            pass

with open('data/generated_data/RPI7317/RPITER_RPI7317_RNA_structure.fasta','w') as f:
    for name, struct in RNA_struct_dict.items():
        f.write(">"+name+"\n")
        index = struct.rfind('(')
        f.write(struct[:index]+"\n")