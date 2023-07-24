#!/usr/bin/env python3
#coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from multiprocessing import Pool, Process
import json, jieba, threading

datas = []
labels = []
train_sum = 0

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    return list(stopwords_list)

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

def work(op, vectorizer, classifier):
    X = vectorizer.fit_transform(datas)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=47)
    # X_train, X_test, y_train, y_test = X[:train_sum], X[train_sum:], labels[:train_sum], labels[train_sum:]
    classifier.fit(X_train, y_train)
    # print("train score:", op, classifier.score(X_train, y_train))
    print("test score:", op, classifier.score(X_test, y_test))

get_data()
processs = []

stop_words_file = 'stopwords.txt'
stopwords = get_custom_stopwords(stop_words_file)


processs.append(Process(target = work, args=(0, CountVectorizer(), MultinomialNB(alpha = 0.1))))
processs.append(Process(target = work, args=(1, CountVectorizer(), ComplementNB(alpha = 0.1))))
processs.append(Process(target = work, args=(2, TfidfVectorizer(), RandomForestClassifier())))
processs.append(Process(target = work, args=(3, TfidfVectorizer(), RandomForestClassifier(n_estimators=155, random_state=43))))

for p in processs: p.start()

for p in processs: p.join()