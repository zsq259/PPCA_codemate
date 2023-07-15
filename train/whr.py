#!/usr/bin/env python3
#coding=utf-8
import jieba
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = []
with open('test.jsonl', 'r',encoding='utf-8') as file:
    for line in file:
        record = json.loads(line)
        data.append(record)

# 处理数据
document = []
for record in data:
    question =" ".join(jieba.cut(record['Question']))
    answer = " ".join(jieba.cut(record['Answer']))
    document.append("Question:  "+question+" Answer: "+answer)
    print([question, answer])
stopwords = [
    "的", "了", "和", "呢", "啊", "哦",
    "就", "而", "或", "及", "与", "等",
    "这", "那", "之", "只", "个",
    "是", "在", "很", "有", "我", "你",
    "他", "她", "它", "我们", "你们", "他们",
    "自己", "什么", "怎么", "为什么", "因为", "所以",
    "如何", "可以", "是否", "是否能够", "能否",
    "是否可以", "能不能", "可以吗", "能不能够", "能否给出",
    "请问", "请教", "请告知", "请帮忙", "请解释",
    "请说明", "请指导", "请提供", "请提醒", "请确认",
    "请回答", "请说一下", "请描述", "请列举", "请比较",
    "请分析", "请解决", "请评价", "请推荐", "请指出",
    "请给出", "请阐述", "请讨论", "请注意", "请考虑",
    "谢谢","求求","高手","感激不尽","请"
    # 其他停用词...
]
tfidf_model_X = TfidfVectorizer(stop_words=stopwords,token_pattern=r"(?u)\b\w+\b").fit_transform(document)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# text = "因为列表是引用数据类型，这是浅复制。浅复制分析：python列表的浅复制对于列表中存在可变的可迭代对象如列表，集合，字典这样的存在也是引用的原对象的地址空间，所以会一同改变。对于列表中存在的数值型数据浅复制会直接创建新的地址空间用以保存。"
# seg_generator = jieba.cut(text)

# for seg in seg_generator:
#     print(seg)