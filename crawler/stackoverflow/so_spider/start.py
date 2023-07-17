#!/usr/bin/env python3
from scrapy import cmdline
from keywords import keywords

key = keywords[0]
command = ["scrapy", "crawl", "stackoverflow", "-a", "knowledge_point={}".format(key)]
cmdline.execute(command)
# cmdline.execute('echo "{}" | scrapy crawl stackoverflow'.split(' ').format(key))