#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get detail web pages in CSDN with threading

import time, html, json, jsonlines, threading
from lxml import etree
from bs4 import BeautifulSoup

from playwright.sync_api import Playwright, sync_playwright, expect

def traverse(s):
    s = html.unescape(s)
    soup = BeautifulSoup(s, 'html.parser')
    text = soup.get_text()
    return text


def run(playwright: Playwright, detail_url) -> None:
    
    global cnt1, cnt2, urls_success
    lock.acquire()
    cnt1 += 1
    lock.release()
    # browser = playwright.chromium.launch(headless=False)
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto(detail_url)
    time.sleep(1)
    tree = etree.HTML(page.content())
    title = tree.xpath("//section[@class='title-box']/h1/text()")[0]
    question = tree.xpath('//section[@class="question_show_box"]//div[@class="md_content_show"]//text()')
    answer = tree.xpath("(//div[@class='answer_box']//div[@class='answer-content-item'])[1]//text()")
    answer = '\n'.join(answer)
    answer = traverse(answer)
    title = traverse(title)
    question = ('\n').join(question)
    question = ('\n').join([title, question])
    question = traverse(question)
    QA_ = {"Answer": answer, "Konwledge_Point": '', "Question": question, "Tag": ''}
    with open("./CSDN_精选/success.out", 'a') as f:
        f.write(detail_url)
    with jsonlines.open("./CSDN_精选/data.jsonl", 'a') as writer:
        writer.write(QA_)
    lock.acquire()
    urls_success.add(detail_url)
    cnt2 += 1
    lock.release()
    page.close()
    # ---------------------
    context.close()
    browser.close()


def work(detail_url) -> None:
    global pool_sema
    if detail_url in urls_success: 
        pool_sema.release()
        return
    with sync_playwright() as playwright:
        run(playwright, detail_url)
    pool_sema.release()

urls_success = set()
with open("./CSDN_精选/success.out", 'r') as f:
    for line in f: urls_success.add(str(line))


cnt1 = 0
cnt2 = 0


ques = []
with open("111.out", 'r') as f:
    for line in f:
        ques.append(str(line))

print(len(ques))

# if __name__ == '__main__':

lock = threading.Lock()
max_connections = 5
pool_sema = threading.BoundedSemaphore(max_connections)

threads = [threading.Thread(target=work, args=[ques[o]]) for o in range(0, len(ques))]

for t in threads:
    pool_sema.acquire()
    t.start()
for t in threads: t.join()

print(cnt1, cnt2)