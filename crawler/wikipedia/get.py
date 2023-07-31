#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keywords import keywords
import os

file = "./wikipedia_算法分析"

for root, dirs, files in os.walk(file):
    for file in files:
        path = os.path.join(root, file)
        # print(path)
        with open(path, 'r') as f:
            with open("./all_datas.jsonl", 'a') as w:
                for line in f: w.write(line)


