#!/usr/bin/env python3
#coding=utf-8

from multiprocessing import Pool, Process
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import json, torch

train_sum = 0
dataset = []
testset = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file(file_name, tag, op):
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            sent_words = "Question: " + ' '.join(record['Question'])
            sent_words += "Answer: " + ' '.join(record['Answer'])
            sent_words.replace('```\n', '[CODE]')
            if op: dataset.append([sent_words, tag])
            else: testset.append([sent_words, tag])

def get_data():
    global train_sum
    # get_file("datas/CSDN1.jsonl", 1)
    get_file("datas/highQualityTrain.jsonl", 1, 1)
    get_file("datas/lowQualityTrain.jsonl", 0, 1)
    train_sum = len(dataset)
    get_file("datas/highQualityTest.jsonl", 1, 0)
    get_file("datas/lowQualityTest.jsonl", 0, 0)


def collate(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True, 
                                   padding='max_length',
                                   max_length=512,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


get_data()
token = BertTokenizer.from_pretrained('algolet/bert-large-chinese')
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=32,
                                     collate_fn=collate,
                                     shuffle=True,
                                     drop_last=True,
                                     pin_memory=True)

pretrained = BertModel.from_pretrained('algolet/bert-large-chinese').to(device)
for param in pretrained.parameters():
    param.requires_grad_(True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)

        return out


model = Model().to(device)
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        print(i, loss.item(), accuracy)

def test():
    model.eval()
    correct = 0
    total = 0
    loader_test = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=32,
                                              collate_fn=collate,
                                              shuffle=True,
                                              drop_last=True)
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)


test()

