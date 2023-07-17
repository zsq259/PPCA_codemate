#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from keywords import keywords

def get(o):
    print(o)
    with open("links/{}_links.out".format(o), 'a') as writer:
        try:
            with open("detail_url/{}_detail_url_info.jsonl".format(o), 'r') as f:
                for line in f:
                    writer.write(line.split('\"')[1].split('?')[0] + '\n')
        except:
            return

for o in keywords: get(o)