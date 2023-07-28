#!/usr/bin/env python3
#coding=utf-8
from sklearn.base import BaseEstimator, ClassifierMixin

from multiprocessing import Pool, Process
from transformers import BertTokenizer, BertModel, AutoModel
from torch.optim import AdamW
import json, torch, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained('bert-base-chinese').to(device)
        self.fc1 = torch.nn.Sequential( torch.nn.Linear(768, 1600), torch.nn.BatchNorm1d(1600), torch.nn.ReLU(True))
        self.fc2 = torch.nn.Sequential( torch.nn.Linear(1600, 800), torch.nn.BatchNorm1d(800), torch.nn.ReLU(True))
        self.fc3 = torch.nn.Sequential( torch.nn.Linear(800, 200), torch.nn.BatchNorm1d(200), torch.nn.ReLU(True))
        self.fc4 = torch.nn.Linear(200, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out = out.last_hidden_state[:, 0]
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.softmax(dim=1)
        return out

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, parameter1=1, parameter2='default'):
        # 初始化分类器的参数
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.model = Model()
        self.model.load_state_dict(torch.load(parameter2))
        self.token = BertTokenizer.from_pretrained('bert-base-chinese')

    def collate(self, sents):
        st = random.randint(0, max(0, len(sents) - 512 - 1))
        sents, labels = sents[st:], labels[st:]
        data = self.token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                    truncation=True,
                                    #    padding=False,
                                    padding='max_length',
                                    max_length=512,
                                    return_tensors='pt',
                                    return_length=True)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        labels = torch.LongTensor(labels)
        return input_ids, attention_mask, token_type_ids, labels


    def predict(self, X):
        datas = [ "Question: " + record['Question'] + " Answer: " + record['Answer'] for record in X]
        predictions = []  # 存储预测结果的列表
        self.model.eval()
        loader_test = torch.utils.data.DataLoader(dataset=datas,
                                                batch_size=32,
                                                collate_fn=self.collate,
                                                shuffle=True,
                                                drop_last=True)
        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(loader_test):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            out = out.argmax(dim=1)
            for op in out: predictions.append(op)
        return predictions