import linecache
from pydoc import doc
from torch.utils.data import Dataset
import re
import random
import numpy as np
from numpy.linalg import norm
from transformers import BertTokenizer

def cosine_sim(a,b):
    return np.dot(a, b) / (norm(a) * norm(b))

class ContrasDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer,emb_path='../embeddings/music_data',aug_strategy=["sent_deletion", "term_deletion", "qd_reorder"]):
        super(ContrasDataset, self).__init__()
        self._filename = filename  # train_data in runbertcontras.py
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._aug_strategy = aug_strategy
        self._rnd = random.Random(0)
        self.product_attr = dict()
        self.product_set = set()
        with open(filename, "r") as f:
            self._total_data = 0
            for line in f.readlines():
                self._total_data += 1
                l = (line.strip()).split('\t')
                for i1 in range(len(l)):
                    if ((i1%3)==0) and (i1>0):
                        if l[i1-2] not in self.product_set:
                            self.product_attr[l[i1-2]] = (l[i1].strip()).replace('  ',' ')
                            self.product_set.add(l[i1-2])
                    else:
                        continue
        # print(self.product_attr['B00GTGQ0PI'])
        self.product_emb_path = emb_path + '/product_emb_out.txt'
        self.query_dict_path = emb_path+'/query_dict.txt'
        self.user_emb_path = emb_path+'/user_emb_out.txt'
        self.word_emb_path = emb_path+'/word_emb_out.txt'
        self.product_emb_dict = dict() # key is the id, and value is the embedding vector
        self.product_list = []

        with open(self.product_emb_path,"r") as f1:
            for line in f1.readlines():
                l = (line.strip()).split('\t',2)
#                 if l[0] in self.product_set:
                self.product_emb_dict[l[0]] = l[-1].split(' ')
                self.product_list.append(l[0])
        # print(type(self.product_emb_dict['B0002E1NQ4']),self.product_emb_dict['B0002E1NQ4'])
        self.query_idx = dict()
        self.query_dict = dict()
        with open(self.query_dict_path,"r") as f2:
            cnt = 0
            for line in f2.readlines():
                l = line.strip().split('\t',2)
                query = l[0].replace("'",'')
                self.query_idx[query.replace('+','')] = cnt
                self.query_dict[cnt] = l[-1].split(' ')
                cnt += 1
        # print(type(self.query_dict[1]),self.query_dict[1])

        self.word_emb_dict = dict()
        with open(self.word_emb_path,"r") as f3:
            for line in f3.readlines():
                l = line.strip().split('\t')
                self.word_emb_dict[l[0]]  = l[-1].split(' ')
        # print(type(self.word_emb_dict['old']),self.word_emb_dict['old'])
        self.user_emb_dict = dict()
        with open(self.user_emb_path,"r") as f4:
            for line in f4.readlines():
                l = line.strip().split('\t')
                self.user_emb_dict[l[0]] = l[-1].split(' ')
        # print(type(self.user_emb_dict['AJGD0VSCUJUP5']),self.user_emb_dict['AJGD0VSCUJUP5'])
#         product_num = len(self.product_emb_dict)
#         self.product_mat = np.zeros((product_num,product_num))
#         print("total size of matrix:{}".format(str(product_num)))
#         for i in range(product_num):
#             for j in range(i, product_num):
#                 i_emb = self.product_emb_dict[self.product_list[i]]
#                 j_emb = self.product_emb_dict[self.product_list[j]]
#                 i_emb = np.array(i_emb,dtype=float)
#                 j_emb = np.array(j_emb,dtype=float)
#                 self.product_mat[i][j] = self.product_mat[j][i] = np.dot(i_emb, j_emb) / (norm(i_emb) * norm(j_emb))
#                 self.product_mat[j][i] = self.product_mat[i][j]
#                 print(self.product_mat[j][i])
#             if (i%1000) ==0 and (i > 0):
#                 print("current finish:{}".format(str((i+1)/product_num)))
        #     break
        print("matrix finish!")

    def check_length(self, pairlist):
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            # line 92-93 is added
            if len(pairlist) == 1:
                return pairlist
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist

    def main_encode(self, qd_pairs):
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        # print(len(qd_pairs),len(all_qd))
        all_qd = self.check_length(all_qd)
        # print(all_qd)
        try:
            history = all_qd[:-2]
            query_tok = all_qd[-2]
            doc_tok = all_qd[-1]
            history_toks = ["[CLS]"]
            segment_ids = [0]
            for iidx, sent in enumerate(history):
                history_toks.extend(sent + ["[eos]"])
                segment_ids.extend([0] * (len(sent) + 1))
            query_tok += ["[eos]"]
            query_tok += ["[SEP]"]
            doc_tok += ["[eos]"]
            doc_tok += ["[SEP]"]
            all_qd_toks = history_toks + query_tok + doc_tok
            segment_ids.extend([0] * len(query_tok))
            segment_ids.extend([0] * len(doc_tok))
            all_attention_mask = [1] * len(all_qd_toks)
            assert len(all_qd_toks) <= self._max_seq_length
            while len(all_qd_toks) < self._max_seq_length:
                all_qd_toks.append("[PAD]")
                segment_ids.append(0)
                all_attention_mask.append(0)
            assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
            anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
            input_ids = np.asarray(anno_seq)
            all_attention_mask = np.asarray(all_attention_mask)
            segment_ids = np.asarray(segment_ids)
            return input_ids, all_attention_mask, segment_ids
        except Exception as e:
            if len(all_qd)==1:
                history = all_qd[:-1]
                query_tok = all_qd[0]
                doc_tok = all_qd[0]
            else:
                try:
                    history = all_qd[:-1]
                    query_tok = all_qd[-1]
                    doc_tok = all_qd[0]
                except Exception as e:
                    print(e)
                    print(qd_pairs)
                    print(all_qd)
                    history_toks = query_tok= doc_tok=all_qd
                
            history_toks = ["[CLS]"]
            segment_ids = [0]
                # line 140 - line 156 is for dataset "Cell phone"
            if len(query_tok) + len(doc_tok) + len(history_toks) >= 120:
                query_tok_new = []
                for q in query_tok:
                    if ("#" in q) or (len(q)<3) :
                        continue
                    else:
                            query_tok_new.append(q)
                    if len(query_tok_new) == 0:
                        for q in query_tok[:8]:
                            query_tok_new.append(q)
                    doc_tok_new = []
                    for d in doc_tok:
                        if ("#" in d) or (len(d) <3 ):
                            continue
                        else:
                            doc_tok_new.append(d)
                    if len(doc_tok_new) == 0:
                        for d in doc_tok[:8]:
                            doc_tok_new.append(d)
                if (len(history_toks) + len(query_tok_new) + len(doc_tok_new)) > 124:
                    query_tok = query_tok_new[:int(len(query_tok_new)/2)]
                    doc_tok = doc_tok_new[:int(len(doc_tok_new)/2)]
                else:
                    query_tok = query_tok_new
                    doc_tok = doc_tok_new
            for iidx, sent in enumerate(history):
                if (len(history_toks) + len(query_tok) + len(doc_tok)) >= 120:
                    break
                else:
                    history_toks.extend(sent + ["[eos]"])
                    segment_ids.extend([0] * (len(sent) + 1))
            query_tok += ["[eos]"]
            query_tok += ["[SEP]"]
            doc_tok += ["[eos]"]
            doc_tok += ["[SEP]"]
            all_qd_toks = history_toks + query_tok + doc_tok
            # print(min_left,len(all_qd_toks))
            segment_ids.extend([0] * len(query_tok))
            segment_ids.extend([0] * len(doc_tok))
            all_attention_mask = [1] * len(all_qd_toks)
            if(len(all_qd_toks) > self._max_seq_length):
                print(len(all_qd_toks),len(segment_ids) ,len(all_attention_mask) ,self._max_seq_length,end = '|')
            # assert len(all_qd_toks) <= self._max_seq_length
            # print(len(all_qd_toks),len(segment_ids) ,len(all_attention_mask) ,self._max_seq_length,end = '|')
            while len(all_qd_toks) < self._max_seq_length:
                all_qd_toks.append("[PAD]")
                segment_ids.append(0)
                all_attention_mask.append(0)
            # print(len(all_qd_toks),len(segment_ids) ,len(all_attention_mask) ,self._max_seq_length)
            # print(len(all_qd_toks),len(segment_ids),len(all_attention_mask),self._max_seq_length)
            assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
            anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
            input_ids = np.asarray(anno_seq)
            all_attention_mask = np.asarray(all_attention_mask)
            segment_ids = np.asarray(segment_ids)
            return input_ids, all_attention_mask, segment_ids

    def _term_deletion(self, sent, ratio=0.5):
        tokens = sent.split()
        num_to_delete = int(round(len(tokens) * ratio))
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        self._rnd.shuffle(cand_indexes)
        output_tokens = list(tokens)
        deleted_terms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(deleted_terms) >= num_to_delete:
                break
            if len(deleted_terms) + len(index_set) > num_to_delete:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_token = "[term_del]"
                output_tokens[index] = masked_token
                deleted_terms.append((index, tokens[index]))
        assert len(deleted_terms) <= num_to_delete
        return " ".join(output_tokens)

    def augmentation(self, sequence, strategy, order = 1, i_qd = [],cur_product=[]):
        random_positions = -1
        # sequence len: num of augmentation product * 2
        cur_query_set = []
        cur_attr_set = []
        max_len = 0
        # print(strategy)
        for i in range(len(sequence)):
            if i % 2==0:
                seq_tmp = sequence[i].strip()
                idx = self.query_idx[seq_tmp.replace('  ',' ')]
                tmp = [int(k) for k in self.query_dict[idx]]
                cur_query_set.append(tmp)
                if len(tmp) > max_len:
                    max_len = len(tmp)
            else:
                cur_attr_set.append(sequence[i])

        query_set = []
        for item in cur_query_set:
            new_item = item
            if len(item) < max_len:
                average = np.mean(np.array(item))
                for j in range(max_len-len(item)):
                    new_item.append(int(average))
            query_set.append(new_item)

        if strategy == "sent_deletion":
            random_num = int(len(sequence) * 0.5)
            random_positions = self._rnd.sample(list(range(len(sequence))), random_num)
            for random_position in random_positions:
                sequence[random_position] = "[sent_del]"
            aug_sequence = sequence

        elif strategy == "term_deletion": # random_positions=-1
            # delete terms (might be different) in 4(or 2) sentences
            aug_sequence = []
            for sent in sequence:
                sent_aug = self._term_deletion(sent)
                sent_aug += " "
                sent_aug = re.sub(r'(\[term_del\] ){2,}', "[term_del] ", sent_aug)
                sent_aug = sent_aug[:-1]
                aug_sequence.append(sent_aug)

        elif strategy=='qd_reorder':
            change_pos = self._rnd.sample(list(range(len(sequence) // 2)), 2)
            aug_sequence = sequence.copy()
            tmp = sequence[change_pos[1] * 2:change_pos[1] * 2 + 2]
            aug_sequence[change_pos[1] * 2:change_pos[1] * 2 + 2] = sequence[change_pos[0] * 2:change_pos[0] * 2 + 2]
            aug_sequence[change_pos[0] * 2:change_pos[0] * 2 + 2] = tmp

        elif strategy=='query_del':
            product_num = int(len(sequence) / 2)
            min_sim = 100.0
            random_positions = 0
            cur_query = product_num - 1
            aug_sequence = []
            random_positions = []
            if order == 1:
                for i in range(product_num):
                    i_score = np.dot(query_set[i], query_set[cur_query]) / (norm(query_set[i]) * norm(query_set[cur_query]))
                    if i_score < min_sim:
                        min_sim = i_score
                        random_positions.append(i*2)
                        random_positions.append(i*2+1)
            else:
                random_num = int(len(sequence) * 0.5)
                random_positions = self._rnd.sample(list(range(len(sequence))), random_num)

            for j in range(len(sequence)):
                flag = 0
                for k in random_positions:
                    if j == k:
                        flag = 1
                        break
                if flag == 0:
                    aug_sequence.append(sequence[j])
                else:
                    continue

        elif strategy=='item_replace':
            if len(cur_product) > 2:
                random_positions = random.randint(0, len(cur_product)-1)
            else:
                random_positions = 0
            aug_sequence = []
            if len(self.product_emb_dict) <= len(self.product_attr):
                total_product = len(self.product_emb_dict)
            else:
                total_product = len(self.product_attr)
            before_emb = np.array(cur_product[random_positions],dtype=float)
            change_product = 0
            for i in range(total_product):
                i_emb = np.array(self.product_emb_dict[self.product_list[i]],dtype=float)
                score = np.dot(i_emb, before_emb) / (norm(i_emb) * norm(before_emb))
                if (score < 1.0) and (score > 0.5):
                    change_product = i # the product index of the product
                    if self.product_list[change_product] in self.product_attr.keys():
                        break
                    else:
                        continue
                else:
                    continue
            if change_product ==0 :
                change_product = random.randint(0, total_product)
            change_attr = self.product_attr[self.product_list[change_product]]
            for i in range(len(i_qd)):
                if (i % 3 == 0):
                    continue
                elif (i == (random_positions * 3 + 2)):
                    aug_sequence.append(change_attr)
                else:
                    aug_sequence.append(i_qd[i])
            '''
            aug_sequence = []
            p_idx = 0
            for e in cur_product:
                e_score = 100.0
                min_elem = 0
                cur_score = 100.0
                for j in range(len(sequence)):
                    if (j%3)!=1:
                        continue
                    print(sequence[j])
                    res = cosine_sim(e,self.query_dict[sequence[j].replace('  ',' ')])
                    e_score += res
                    if res < cur_score:
                        min_elem = cur_product[j]
                        cur_score = res
                if e_score < min_sim:
                    min_sim = e_score
                    change_pos = p_idx
                    change_to  = min_elem
                p_idx += 1
            for j in range(len(sequence)):
                if j != change_pos:
                    aug_sequence.append(sequence[j])
                else:
                    aug_sequence.append(cur_product[change_to])
            '''
            # what is the range of the items being searched?
            # Do we need to search all products for item-replacing?
        else:
            print(strategy)
            # assert False
        # print("finish augmentation")
        return aug_sequence, random_positions

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        cnt = 0
        item_seq = []
        qd_pairs = []
        for tmp in line[1:]:
            if cnt%3 ==0:
                item_seq.append(self.product_emb_dict[tmp]) # emb
                cnt += 1
                continue
            else:
                cnt += 1
                qd_pairs.append(tmp) #query and attribute
        # print(qd_pairs) -> the original data by line
        random_qd_pairs1 = qd_pairs.copy()
        random_qd_pairs2 = qd_pairs.copy()
        if len(qd_pairs) <= 3: # only contains one query and one clicked document
            aug_strategy = ["sent_deletion", "term_deletion", "query_del"]
        else:
            aug_strategy = self._aug_strategy

        strategy1 = self._rnd.choice(aug_strategy)
        random_qd_pairs1, random_pos1 = self.augmentation(random_qd_pairs1, strategy1, i_qd = line[1:],cur_product=item_seq)
        strategy2 = self._rnd.choice(aug_strategy)
        random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy2,order=2, i_qd = line[1:],cur_product=item_seq)

        while random_pos1 == random_pos2 or strategy1 == strategy2:
            strategy2 = self._rnd.choice(aug_strategy)
            random_qd_pairs2 = qd_pairs.copy()
            random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy2,order=2, i_qd = line[1:],cur_product=item_seq)

        # use this to change the word into encoding
        if random_qd_pairs1 == None:
            print("random_qd_pairs1")
        if random_qd_pairs2 == None:
            print("random_qd_pair2")
        # print(len(random_qd_pairs1),strategy1,end='|')
        # print(len(random_qd_pairs2),strategy2,end='|')
        input_ids, attention_mask, segment_ids = self.main_encode(random_qd_pairs1)
        input_ids2, attention_mask2, segment_ids2 = self.main_encode(random_qd_pairs2)

        batch = {
            'input_ids1': input_ids,
            'token_type_ids1': segment_ids,
            'attention_mask1': attention_mask,
            'input_ids2': input_ids2,
            'token_type_ids2': segment_ids2,
            'attention_mask2': attention_mask2,
        }

        return batch

    def __len__(self):
        return self._total_data

if __name__ =='__main__':
    # filepath = './data/Musical_Instruments/data2.txt'
    # filepath = '/Users/tt/Downloads/COCA/ContrastiveLearning/data/aol/train.pos.sample.txt'
    filepath = "/home/shitong_dai/seqpps/pre_train/data/Clothing_Shoes_and_Jewelry/Clothing_Shoes_and_Jewelry/train_data.txt"
    # filepath = '/Users/tt/Downloads/seqPPS/pre_train/data/Clothing_Shoes_and_Jewelry/train_data.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tmp = "item_replace,sent_deletion"
    aug_strategy = tmp.split(",")
    # train_dataset = ContrasDataset(filepath, 128, tokenizer,emb_path='/Users/tt/Downloads/seqPPS/embeddings/clothes_data',aug_strategy=aug_strategy)
    train_dataset = ContrasDataset(filepath, 128, tokenizer,aug_strategy=aug_strategy)

    batch = train_dataset.__getitem__(162)
    print(batch)
    # print(batch['input_ids1'])
    # print(batch['attention_mask1'])
    # print(batch['token_type_ids1'])
    # print(batch['input_ids2'])
    # print(batch['attention_mask2'])
    # print(batch['token_type_ids2'])
