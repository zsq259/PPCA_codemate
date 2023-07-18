#!/usr/bin/env python3
from multiprocessing import Pool, Process
from scrapy import cmdline
from keywords import keywords


def work(key):
    command = ["scrapy", "crawl", "stackoverflow", "-a", "knowledge_point={}".format(key)]
    cmdline.execute(command)

processs = [Process(target = work, args=[key]) for key in keywords]

for process in processs:
    process.start()
    process.join()
    