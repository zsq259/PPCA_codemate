#!/usr/bin/env python3
#coding=utf-8

from multiprocessing import Pool, Process
from transformers import BertTokenizer, BertModel, AutoModel
from torch.optim import AdamW
import json, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
