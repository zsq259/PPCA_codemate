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
        self.fc1 = torch.nn.Sequential( torch.nn.Conv1d(1, 16, kernel_size=3,padding=1),torch.nn.BatchNorm1d(16), torch.nn.ReLU(True) )
        self.fc2 = torch.nn.Sequential( torch.nn.Conv1d(16, 64, kernel_size=3,padding=1),torch.nn.BatchNorm1d(64), torch.nn.ReLU(True) )
        self.fc3 = torch.nn.Sequential( torch.nn.Conv1d(64, 256, kernel_size=3,padding=1),torch.nn.BatchNorm1d(256), torch.nn.ReLU(True) )
        self.fc4 = torch.nn.Sequential( torch.nn.Linear(256*768, 2), torch.nn.BatchNorm1d(2), torch.nn.ReLU(True))

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        out = out.last_hidden_state[:, 0]
        out = out.view(out.shape[0],-1,768)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(out.shape[0],-1)
        out = self.fc4(out)
        out = out.softmax(dim=1)
        return out
