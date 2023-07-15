#!/usr/bin/env python3
#coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import json, jieba, threading
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

def work(vectorizer, op):
    X = vectorizer.fit_transform(datas)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=43)
    # X_train, X_test, y_train, y_test = X[:train_sum], X[train_sum:], labels[:train_sum], labels[train_sum:]
    rf_classifier = RandomForestClassifier(n_estimators = 55, random_state = 43)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    # 进行性能评估，如准确率、精确率、召回率等
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(op, accuracy, precision)

get_data()

thread1 = threading.Thread(target = work, args = [TfidfVectorizer(), 0])
thread2 = threading.Thread(target = work, args = [TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"), 1])
thread3 = threading.Thread(target = work, args = [TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words = stopwords.stopwords, max_df=0.6), 2])

thread1.start()
thread2.start()
thread3.start()

thread1.join()
thread2.join()
thread3.join()
# print(vectorizer.vocabulary_)

