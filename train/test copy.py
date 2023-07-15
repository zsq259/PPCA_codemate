#!/usr/bin/env python3
#coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sentence_transformers import SentenceTransformer,util
import json, jieba, threading
import stopwords

datas = []
labels = []

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
    get_file("datas/highQualityTrain.jsonl", 1)
    get_file("datas/lowQualityTrain.jsonl", 0)
    get_file("datas/highQualityTest.jsonl", 1)
    get_file("datas/lowQualityTest.jsonl", 0)

    

get_data()

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
X = model.encode(datas)
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = X[:9738], X[9738:], labels[:9738], labels[9738:]
rf_classifier = RandomForestClassifier(n_estimators = 155, random_state = 43)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
# 进行性能评估，如准确率、精确率、召回率等
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(accuracy, precision)

