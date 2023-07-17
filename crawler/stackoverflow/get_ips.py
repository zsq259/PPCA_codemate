#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml, json
with open ("ips.yaml") as f:
    data = yaml.safe_load(f)
    # print(data)
    record = data['proxies']
    print(record)
    