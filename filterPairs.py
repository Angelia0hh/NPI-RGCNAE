#为RPITER 7317生成pairs
base_path = 'data/generated_data/RPI7317/'
def read_pairs(path):
    pairs = []
    with open(path)as f:
        for line in f:
            protein, RNA = line.rstrip().split('_')
            if RNA not in ['KCNQ1OT1', 'CCDC26', 'DLX6-AS1', 'FTX', 'TSIX']:
                pairs.append([protein, RNA])
    return pairs


positive_pairs = read_pairs(base_path+'RPI7317_positive_pairs.txt')
random_negative_pairs = read_pairs(base_path+'RPI7317_random_negative_pairs.txt')
sort_negative_pairs = read_pairs(base_path+'RPI7317_sort_negative_pairs.txt')
sort_random_negative_pairs = read_pairs(base_path+'RPI7317_sort_random_negative_pairs.txt')

print(len(positive_pairs))
print(len(random_negative_pairs))
print(len(sort_negative_pairs))
print(len(sort_random_negative_pairs))

with open('data/generated_data/RPI7317/RPITER_RPI7317_sort_pairs.txt','w')as f:
    for pair in positive_pairs:
        f.write(pair[0]+"\t"+pair[1]+"\t"+"1\n")
    for pair in sort_negative_pairs:
        f.write(pair[0]+"\t"+pair[1]+"\t"+"0\n")

with open('data/generated_data/RPI7317/RPITER_RPI7317_random_pairs.txt','w')as f:
    for pair in positive_pairs:
        f.write(pair[0] + "\t" + pair[1] + "\t" + "1\n")
    for pair in random_negative_pairs:
        f.write(pair[0] + "\t" + pair[1] + "\t" + "0\n")

with open('data/generated_data/RPI7317/RPITER_RPI7317_sort_random_pairs.txt','w')as f:
    for pair in positive_pairs:
        f.write(pair[0] + "\t" + pair[1] + "\t" + "1\n")
    for pair in sort_random_negative_pairs:
        f.write(pair[0] + "\t" + pair[1] + "\t" + "0\n")

