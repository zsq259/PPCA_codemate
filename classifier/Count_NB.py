#!/usr/bin/env python3
#coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from multiprocessing import Pool, Process
import json, jieba, threading

datas = []
labels = []
train_sum = 0

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    # custom_stopwords_list = [i for i in stopwords_list]
    # print(custom_stopwords_list)
    return list(stopwords_list)

def get_file(file_name, tag):
    global datas, labels
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            data_words = "Question: " + ' '.join(jieba.cut(record['Question']))
            data_words += "Answer: " + ' '.join(jieba.cut(record['Answer']))
            datas.append(data_words)
            labels.append(tag)

def get_data():
    global train_sum
    get_file("datas/CSDN.jsonl", 1)
    get_file("datas/highQualityTrain.jsonl", 1)
    get_file("datas/lowQualityTrain.jsonl", 0)
    train_sum = len(datas)
    get_file("datas/highQualityTest.jsonl", 1)
    get_file("datas/lowQualityTest.jsonl", 0)

def work(op, vectorizer, nb_classifier):
    X = vectorizer.fit_transform(datas)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=47)
    # X_train, X_test, y_train, y_test = X[:train_sum], X[train_sum:], labels[:train_sum], labels[train_sum:]
    nb_classifier.fit(X_train, y_train)
    train_score = nb_classifier.score(X_train, y_train)
    print("train score:", op, nb_classifier.score(X_train, y_train))
    print("test score:", op, _classifier.score(X_test, y_test))

get_data()
processs = []

stop_words_file = 'stopwords.txt'
stopwords = get_custom_stopwords(stop_words_file)



processs.append(Process(target = work, args=(0, CountVectorizer(), MultinomialNB())))
processs.append(Process(target = work, args=(1, CountVectorizer(max_df = 0.8, min_df = 3, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', stop_words=list(stopwords)), MultinomialNB())))
# processs.append(Process(target = work, args = [TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"), 2]))
# processs.append(Process(target = work, args = [TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words = stopwords.stopwords, max_df=0.6), 3]))
# processs.append(Process(target = work, args=(0, CountVectorizer(), MultinomialNB(alpha=1.0))))
# processs.append(Process(target = work, args=(1, CountVectorizer(), MultinomialNB(alpha = 0.1))))
# processs.append(Process(target = work, args=(2, CountVectorizer(), ComplementNB())))

for p in processs: p.start()

for p in processs: p.join()