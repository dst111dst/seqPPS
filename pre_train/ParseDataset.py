import os
import csv
import pandas as pd

def filter_words(s):
    ans = s.replace('[', '')
    ans = ans.replace(']', '')
    ans = ans.replace("'", '')
    return ans.split(',')

data_dir = '/Users/tt/Downloads/cl4pps/data/processed'
dataset_name = 'Musical_Instruments'
data_dir = data_dir +'/'+dataset_name +'/'
# full_df= pd.read_csv(os.path.join(data_dir, 'full.csv'))

user_dict = dict()
user_set = set()
with open(os.path.join(data_dir, 'full.csv'),"r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    user_count = 0
    data_list = list(csv_reader)
    for row in data_list[1:]:
        user = row[0]
        if user not in user_set:
            user_set.add(user)
            user_dict[user] = row[4]+'\t'+row[1]+'\t'+row[2] # query \t asin \t reviewText
        else:
            user_dict[user] += '\t'+ row[4]+'\t'+row[1]+'\t'+row[2]
        # print(type(user_dict[user]))
        # break

save_dir = '/Users/tt/Downloads/seqPPS/pre_train/data'
save_dir = save_dir +'/'+dataset_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open("{}/data.txt".format(save_dir),'w')as fin:
    for k,v in user_dict.items():
        fin.write(k)
        value = v.split('\t')
        if len(value) <4: # contain only one product
            fin.write('\t')
            query = filter_words(value[0])
            for word in query[:-1]:
                fin.write(word)
                fin.write(' ')
            fin.write(query[-1])
            fin.write('\t')
            review = filter_words(value[2])
            for word in review[:-1]:
                fin.write(word)
                fin.write(' ')
            fin.write(review[-1])

        else:
            iter_num = len(value)//3
            # print(len(value),iter_num)
            for i in range(iter_num):
                fin.write('\t')
                query = filter_words(value[0+i*3])
                for word in query[:-1]:
                    fin.write(word)
                    fin.write(' ')
                fin.write(query[-1])
                fin.write('\t')
                review = filter_words(value[2+i*3])
                for word in review[:-1]:
                    fin.write(word)
                    fin.write(' ')
                fin.write(review[-1])
        fin.write('\n')



