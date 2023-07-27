#!/usr/bin/env python3
#coding=utf-8

from multiprocessing import Pool, Process
from transformers import BertTokenizer, BertModel, AutoModel
from torch.optim import AdamW
import json, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_sum = 0
dataset = []
testset = []
test_accuracy = []

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
#     get_file("datas/CSDN1.jsonl", 1)
    # get_file("../input/dataset/datas/highQualityTrain.jsonl", 1, 1)
    # get_file("../input/dataset/datas/lowQualityTrain.jsonl", 0, 1)
    # train_sum = len(dataset)
    # get_file("../input/dataset/datas/highQualityTest.jsonl", 1, 0)
    # get_file("../input/dataset/datas/lowQualityTest.jsonl", 0, 0)
    get_file("datas/highQualityTrain.jsonl", 1, 1)
    get_file("datas/lowQualityTrain.jsonl", 0, 1)
    train_sum = len(dataset)
    get_file("datas/highQualityTest.jsonl", 1, 0)
    get_file("datas/lowQualityTest.jsonl", 0, 0)



def work(tokenizer, model_name, batch_size_, max_length, requires_grad_op, learning_rate, weight_decay_):
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

    model = Model().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    def collate(data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]
        data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                    truncation=True,
                                    #    padding=False,
                                    padding='max_length',
                                    max_length=max_length,
                                    return_tensors='pt',
                                    return_length=True)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        labels = torch.LongTensor(labels)
        return input_ids, attention_mask, token_type_ids, labels

    def train(cnt_num):
        print("start {}-th train".format(cnt_num))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size_,
                                        collate_fn=collate,
                                        shuffle=True,
                                        drop_last=True,
                                        pin_memory=True)
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
            if i % 50 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                print(i, loss.item(), accuracy)


    token = BertTokenizer.from_pretrained(tokenizer)
    pretrained = AutoModel.from_pretrained(model_name).to(device)
    for param in pretrained.parameters():
        param.requires_grad_(requires_grad_op)

    def test(cnt_num):
        model.eval()
        correct = 0
        total = 0

        loader_test = torch.utils.data.DataLoader(dataset=testset,
                                                batch_size=batch_size_,
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

        print("test {} score:".format(cnt_num), correct / total)
        test_accuracy.append(correct / total)

    for i in range(0, 100):
        train(i)
        test(i)
    save_path = "./model.pt"
    torch.save(model.state_dict(), save_path)

get_data()
work('bert-base-chinese', "bert-base-chinese", 32, 512, False, 5e-5, 1e-5)
print(test_accuracy)


