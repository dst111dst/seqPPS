import os
import random

def filter_words(s):
    ans = s.replace('[', '')
    ans = ans.replace(']', '')
    ans = ans.replace("'", '')
    return ans.split(',')

dataset_name = 'Musical_Instruments'
data_dir = '/Users/tt/Downloads/seqPPS/pre_train/data'
data_dir = data_dir +'/'+dataset_name

save_dir = '/Users/tt/Downloads/seqPPS/fine_tune/data'
save_dir= save_dir +'/'+dataset_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_path = '/Users/tt/Downloads/seqPPS/pre_train/data/Musical_Instruments/data.txt'
file_list = []
with open(file_path,'r') as fin:
    for line in fin.readlines():
        l = line.split('\t')
        content = '1'
        for s in l[1:]:
            content = content + '\t' + s
        cnt = len(l) - 1  # the product number in a user-log
        file_list.append(content)
        if cnt == 2:
            content = '0\t'
            review = filter_words(l[1])
            pos1 = random.randint(0,len(review))
            content = content + l[1].replace(review[pos1],'') +'\t'
            query = filter_words(l[-1])
            pos2 = random.randint(0,len(query))
            content = content + l[-1].replace(review[pos2],'')
            file_list.append(content)
        else:
            product_num = cnt // 2
            if product_num > 5:
                product_num = 5
            for i in range(0,product_num):
                content = '0'
                tmp = [s for s in l[1:]]
                for idx in range(cnt):
                    if (idx == 2*i) or (idx == 2*i + 1):
                        continue
                    content += '\t'
                    content += tmp[idx]
                file_list.append(content)

with open("{}/data.txt".format(save_dir),'w')as f2:
    for line in file_list:
        f2.write(line)




