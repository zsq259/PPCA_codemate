#!/usr/bin/env python3
#coding=utf-8
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import json, joblib


test_datas = []
test_labels = []

def get_file(file_name, tag):
    global datas, labels
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            test_datas.append(record)
            test_labels.append(tag)

def get_data():
    # get_file("datas/CSDN1.jsonl")
    # get_file("highQualityTrain.jsonl", 1)
    # get_file("lowQualityTrain.jsonl", 0)
    get_file("highQualityTest.jsonl", 1)
    get_file("lowQualityTest.jsonl", 0)

get_data()

classifiers = []
classifiers.append(MyBert(1, '0.pt'))
classifiers.append(MyBert(1, '12_1.pt'))
classifiers.append(MyBert(1, '12_2.pt'))

classifiers.append(MyRFClf("vectorizer_34.pkl", "random_forest_34.pkl"))
classifiers.append(MyRFClf("vectorizer_35.pkl", "random_forest_35.pkl"))
classifiers.append(MyRFClf("vectorizer_36.pkl", "random_forest_36.pkl"))
classifiers.append(MyRFClf("vectorizer_37.pkl", "random_forest_37.pkl"))

# test_datas = test_datas[:32]
# test_labels = test_labels[:32]
pres = []
for i in range(0, len(classifiers)):
    print(i)
    pres.append(classifiers[i].predict(test_datas))

pred = []
for i in range(0, len(test_datas)):
    sum = 0
    for j in range(0, len(classifiers)): sum += pres[j][i]
    pred.append(int(sum > (len(classifiers) // 2)))
accuracy = accuracy_score(test_labels, pred)
print(accuracy)

 