#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keywords import keywords

for key in keywords:
    with open("./CSDN_new_算法分析/{}.jsonl".format(key), 'r') as f:
        with open("./all_datas.jsonl", 'a') as w:
            for line in f: w.write(line)


