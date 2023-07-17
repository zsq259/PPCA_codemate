#!/usr/bin/env python3
from scrapy import cmdline
from keywords import keywords

print(len(keywords))
for key in keywords:
    command = ["scrapy", "crawl", "stackoverflow", "-a", "knowledge_point={}".format(key)]
    cmdline.execute(command)
# cmdline.execute('echo "{}" | scrapy crawl stackoverflow'.split(' ').format(key))