#!/usr/bin/env python3
# -*- coding: utf-8 -*-
urls_success = set()
with open("./CSDN_精选/success.out", 'r') as f:
    for line in f: urls_success.add(str(line))

with open("links.out", 'r') as f:
    for line in f:
        if str(line) not in urls_success: print(line, end='')

