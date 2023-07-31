#!/usr/bin/env python3
#coding=utf-8
from sklearn.base import BaseEstimator, ClassifierMixin

from multiprocessing import Pool, Process
from transformers import BertTokenizer, BertModel, AutoModel
from torch.optim import AdamW
import json, torch, random, joblib, jieba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = AutoModel.from_pretrained('bert-base-chinese').to(device)
token = BertTokenizer.from_pretrained('bert-base-chinese')

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Sequential( torch.nn.Linear(768, 1600), torch.nn.BatchNorm1d(1600), torch.nn.ReLU(True))
        self.fc2 = torch.nn.Sequential( torch.nn.Linear(1600, 800), torch.nn.BatchNorm1d(800), torch.nn.ReLU(True))
        self.fc3 = torch.nn.Sequential( torch.nn.Linear(800, 200), torch.nn.BatchNorm1d(200), torch.nn.ReLU(True))
        self.fc4 = torch.nn.Linear(200, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out = out.last_hidden_state[:, 0]
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.softmax(dim=1)
        return out

class MyBert(BaseEstimator, ClassifierMixin):
    def __init__(self, parameter1=1, parameter2='default'):
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.model = Model().to(device)
        self.model.load_state_dict(torch.load(str(parameter2)))

    def predict(self, X):
        def collate(sents):
            st = random.randint(0, max(0, len(sents) - 512 - 1))
            sents = sents[st:]
            data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=512,
                                        return_tensors='pt',
                                        return_length=True)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            token_type_ids = data['token_type_ids']
            return input_ids, attention_mask, token_type_ids

        datas = [ ("Question: " + ' '.join(record['Question']) + "Answer: " + ' '.join(record['Answer'])).replace('```\n', '[CODE]') for record in X]
        predictions = []
        self.model.eval()
        loader_test = torch.utils.data.DataLoader(dataset=datas,
                                                batch_size=32,
                                                collate_fn=collate,
                                                shuffle=False,
                                                drop_last=False)
        for i, (input_ids, attention_mask, token_type_ids) in enumerate(loader_test):
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            out = out.argmax(dim=1)
            for op in out: predictions.append(op.item())
        return predictions
    

class MyRFClf(BaseEstimator, ClassifierMixin):
    def __init__(self, parameter1="", parameter2='default'):
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.vectorizer = joblib.load(parameter1)
        self.classfier = joblib.load(parameter2)

    def fit(self, X, y):
        pass

    def predict(self, X):
        datas = [("Question: " + ' '.join(jieba.cut(record['Question'])) + " Answer: " + ' '.join(jieba.cut(record['Answer']))).replace('```\n', '[CODE]') for record in X]
        X = self.vectorizer.transform(datas)
        return self.classfier.predict(X)