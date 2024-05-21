from nltk import word_tokenize
from collections import Counter
import jieba
import numpy as np
from torch.autograd import Variable
from transformer.model import subsequent_mask


def data_load(path):
    ens = []
    zhs = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            en, zh = line.split("\t")
            en = ['BOS'] + word_tokenize(en) + ['EOS']
            zh = ['BOS'] + [char for char in jieba.cut(zh.strip())] + ['EOS']
            ens.append(en)
            zhs.append(zh)
    return ens, zhs


def create_dic(sentences, max_len):
    word_count = Counter(word for sentence in sentences for word in sentence)
    pre_dic = word_count.most_common(max_len)
    word_dic = {word[0]: index + 2 for index, word in enumerate(pre_dic)}
    word_dic.update({"UNK": 0, "PAD": 1})
    return word_dic


def word2id(ens, zhs, dic, sort):
    out_ens_id = [[dic.get(word, dic.get("UNK")) for word in sent] for sent in ens]
    out_zhs_id = [[dic.get(word, dic.get("UNK")) for word in sent] for sent in zhs]

    def len_sort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort:
        index_sort = len_sort(out_ens_id)
        out_ens_id = [out_ens_id[index] for index in index_sort]
        out_zhs_id = [out_zhs_id[index] for index in index_sort]
    return out_ens_id, out_zhs_id


def seq_padding(sens):
    batch_max_len = 0
    for sen in sens:
        if batch_max_len < len(sen):
            batch_max_len = len(sen)
    for sen in sens:
        for i in range(batch_max_len - len(sen)):
            sen.append(1)

    return sens

'''
##生成batch
##读取数据并构建Dataset子类
from torch.utils.data import Dataset, DataLoader


##创建模型数据集类
class en2zh_dataSet(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        en = self.datas[0]
        zh = self.datas[1]
        return en, zh


##定义数据读取方法，生成batch
def Batch(datas):
    dataset = en2zh_dataSet(datas)
    return DataLoader(dataset)
'''

import torch
import numpy as np


class Batch:
    def __init__(self, src, trg=None, pad=1):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def split_batch(en, zh, batch_size, shuffle):
    id_list = np.arange(0, len(en), batch_size)
    if shuffle:
        np.random.shuffle(id_list)
    batch_indexes = []
    for batch_index in id_list:
        batch_indexes.append(np.arange(batch_index, min(batch_index + batch_size, len(en))))
    batches = []
    for batch_index in batch_indexes:
        batch_en = [en[index] for index in batch_index]
        batch_zh = [zh[index] for index in batch_index]
        batch_en = seq_padding(batch_en)
        batch_zh = seq_padding(batch_zh)
        batches.append(Batch(torch.Tensor(batch_en).long(), torch.Tensor(batch_zh).long()))
    return batches
