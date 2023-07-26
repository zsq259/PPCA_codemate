#!/usr/bin/env python3
#coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from multiprocessing import Pool, Process
import json, jieba, gensim
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
import stopwords

datas = []
labels = []
train_sum = 0

def get_file(file_name, tag):
    global datas, labels
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            data_words = "Question: " + ' '.join(jieba.cut(record['Question']))
            data_words += "Answer: " + ' '.join(jieba.cut(record['Answer']))
            data_words.replace('```\n', '[CODE]')
            datas.append(data_words)
            labels.append(tag)

def get_data():
    global train_sum
    get_file("datas/CSDN1.jsonl", 1)
    get_file("datas/highQualityTrain.jsonl", 1)
    get_file("datas/lowQualityTrain.jsonl", 0)
    train_sum = len(datas)
    get_file("datas/highQualityTest.jsonl", 1)
    get_file("datas/lowQualityTest.jsonl", 0)

def get_vec(sentence, model):
    tokens = sentence.split()  # 将句子转换为小写，并分割成单词
    filtered_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if len(filtered_tokens) > 0:
        sentence_vector = sum(model.wv[token] for token in filtered_tokens) / len(filtered_tokens)
    else:
        # 处理无法计算句子向量的情况，根据需求返回合适的默认值或标记
        sentence_vector = [0.0] * model.vector_size  # 使用全零向量作为默认值
    return sentence_vector

def work(op):
    lines = []
    for line in datas: #分别对每段分词
        temp = jieba.lcut(line) 
        words = []
        for i in temp:
            #过滤掉所有的标点符号
            i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
            if len(i) > 0:
                words.append(i)
        if len(words) > 0:
            lines.append(words)
    # print(lines[0:5])#预览前5行分词结果
    model = Word2Vec(lines, vector_size = 20, window = 5 , min_count = 3, epochs=7, negative=10, sg=1)
    # print(get_vec(datas[0], model))



    X = [get_vec(sentence, model) for sentence in datas]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=47)
    # X_train, X_test, y_train, y_test = X[:train_sum], X[train_sum:], labels[:train_sum], labels[train_sum:]
    # classifier = RandomForestClassifier(n_estimators = 155, random_state = 43)
    classifier = RandomForestClassifier(n_estimators=155, random_state=43)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # print(op, accuracy, precision)
    print("train score:", op, classifier.score(X_train, y_train))
    print("test score:", op, classifier.score(X_test, y_test))

get_data()
work(0)