#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
from collections import defaultdict
import jieba
import codecs



def build_vocab(vocab_path, yelp_json_path):

    if os.path.exists(vocab_path):
        vocab_file = open(vocab_path, 'rb')
        vocab = pickle.load(vocab_file)
        print("load focab finish!")
    else:
        # 记录每个单词及其出现的频率
        word_freq = defaultdict(int)
        # 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
        with codecs.open(yelp_json_path, 'rb','utf-8') as f:
            for line in f:
                content = line.split('\001')
                if (len(content)>= 3):
                    words = jieba.cut(content[0])
                    for word in words:
                        word_freq[word] += 1
            print("load finished")

        # 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
        vocab = {}
        i = 1
        vocab['UNKNOW_TOKEN'] = 0
        for word, freq in word_freq.items():
            if freq > 3:
                vocab[word] = i
                i += 1

        # 将词汇表保存下来
        with open(vocab_path, 'wb') as g:
            pickle.dump(vocab, g)
            print( len(vocab) ) # 33692
            print ("vocab save finished")

    return vocab

def load_dataset(title_txt_path, max_word_in_sent):
    yelp_data_path = title_txt_path[0:-4] + "_data.pickle"

    vocab_path = title_txt_path[0:-4] + "_vocab.pickle"
    # doc_num = 229907 #数据个数
    if not os.path.exists(yelp_data_path):
        doc_count = len(open(title_txt_path,'rb').readlines())#159591 有行数159576
        label1_to_id = {}
        i = 1
        with codecs.open(title_txt_path,'rb','utf-8') as f:
            for line in f.readlines():
                contList = line.split('\001')
                if(len(contList)>=3):
                    # title = contList[0]
                    firstTag = contList[1]
                    secTag = contList[2]
                    if not firstTag in label1_to_id:
                        label1_to_id[firstTag] = i
                        i += 1
        print(label1_to_id)
        vocab  = build_vocab(vocab_path, title_txt_path)
        num_classes = len(label1_to_id)
        print(num_classes)
        UNKNOWN = 0

        # data_x = np.zeros([doc_num,max_sent_in_doc,max_word_in_sent])
        data_x2 = np.zeros([159576,max_word_in_sent],dtype= int)
        data_y = []

        #将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
        # 不够的补零，多余的删除，并保存到最终的数据集文件之中
        with codecs.open(title_txt_path, 'rb','utf-8') as f:
            m = 0
            for line in f.readlines():
                word_to_index = np.zeros([max_word_in_sent],dtype = int)
                contentList = line.split('\001')
                if(len(contentList)>=3):
                    for j, word in enumerate(jieba.cut(contentList[0])):
                        if j < max_word_in_sent:
                            word_to_index[j] = vocab.get(word, UNKNOWN)
                            print("sentence word", j, word_to_index[j])
                    # data_x[line_index] = doc
                    #print("the ith is :  ",m)
                    data_x2[m] = word_to_index
                    label = label1_to_id.get(contentList[1])
                    #print(label,contentList[1])
                    labels = [0] * num_classes
                    labels[label-1] = 1
                    data_y.append(labels)
                    m += 1
            print("len_data_y",len(data_y),len(data_x2),m)
            # pickle.dump((data_x, data_y), open(yelp_data_path, 'wb'))
            pickle.dump((data_x2,data_y),open(yelp_data_path,'wb'))
            pickle.dump(label1_to_id,open('../data/label12id.pickle','wb'))
            print(len(data_x2)) #229907
    else:
        data_file = open(yelp_data_path, 'rb')
        # data_x, data_y = pickle.load(data_file)
        data_x2,data_y = pickle.load(data_file)
        label1_to_id= pickle.load(open('../data/label12id.pickle','rb'))

    # length = len(data_x)
    length = len(data_x2)

    # train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1:]
    # train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1:]
    indexId = np.random.permutation(np.arange(length))
    data_x2_shuffle = data_x2[indexId]
    data_y_shuffle = np.array(data_y, dtype=int)[indexId]
    train_x, dev_x = data_x2_shuffle[:int(length * 0.8)], data_x2_shuffle[int(length * 0.8) + 1:]
    train_y, dev_y = data_y_shuffle[:int(length * 0.8)], data_y_shuffle[int(length * 0.8) + 1:]

    indexDev= np.random.permutation(np.arange(len(dev_x)))

    dev_x_shuffle = dev_x[indexDev]
    dev_y_shuffle = dev_y[indexDev]

    test_x,val_x = dev_x_shuffle[:int(len(dev_x) * 0.5)], dev_x_shuffle[int(len(dev_x) * 0.5) + 1:]
    test_y,val_y = dev_y_shuffle[:int(len(dev_x) * 0.5)], dev_y_shuffle[int(len(dev_x) * 0.5) + 1:]

    print("train_x", len(train_x), len(train_y))#143618
    print("the end ")
    print("test_x", len(test_x), len(test_y))#15957
    print("val_x", len(val_x), len(val_y))
    return train_x, train_y, test_x, test_y, val_x, val_y, label1_to_id

def iter_batch(x,y,batch_size = 32):
    data_len = len(x)
    num_batch = int((data_len-1)/batch_size) + 1 # 迭代的batch 数

    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i+1) * batch_size,data_len)
        yield x[start_index:end_index],y[start_index:end_index]



if __name__ == '__main__':
    load_dataset("../data/TitleOutput.txt", 50)

