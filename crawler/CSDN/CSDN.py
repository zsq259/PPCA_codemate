#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, jsonlines
import asyncio
import aiohttp, html
from lxml import etree
from bs4 import BeautifulSoup

url_ = "https://so.csdn.net/api/v3/search?q={}&t=ask&p={}&s=0&tm=0&lv=-1&ft=0&l=&u=&ct=-1&pnt=-1&ry=-1&ss=-1&dct=-1&vco=1&cc=-1&sc=-1&akt=-1&art=-1&ca=1&prs=&pre=&ecc=-1&ebc=-1&ia=1&dId=&cl=-1&scl=-1&tcl=-1&platform=pc&ab_test_code_overlap=&ab_test_random_code="



loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
urls = []
f = open("./CSDN_精选/success.json", 'r')
urls_success = set(json.load(f))
cnt1 = 0
cnt2 = 0

async def traverse(s):
    s = html.unescape(s)
    soup = BeautifulSoup(s, 'html.parser')
    text = soup.get_text()
    # print("s: ", s)
    # print("text:", text)
    return text

async def get_detail(detail_url, answer):
    detail_url = detail_url.split('?')[0]
    # print(detail_url)
    if detail_url in urls_success:
        return
    global cnt1, cnt2
    cnt1 = cnt1 + 1
    async with aiohttp.ClientSession() as session:
        async with session.get(detail_url) as response:
            tree = etree.HTML(await response.text())
            if (tree.xpath("count(//section[@class='title-box']/h1/text())")) < 1:
                print(detail_url)
                return
            title = tree.xpath("//section[@class='title-box']/h1/text()")[0]
            question = tree.xpath('//section[@class="question_show_box"]//div[@class="md_content_show"]//text()')
            answer = await traverse(answer)
            title = await traverse(title)
            question = ('\n').join(question)
            question = ('\n').join([title, question])
            question = await traverse(question)
            QA_ = {"Answer": answer, "Konwledge_Point": '', "Question": question, "Tag": ''}
            urls_success.add(detail_url)
            cnt2 = cnt2 + 1
            with jsonlines.open("./CSDN_精选/data.jsonl", 'a') as writer:
                writer.write(QA_)
    
ques = []
with open("links.out", 'r') as f:
    for line in f:
        ques.append(str(line))

Tasks = [asyncio.ensure_future(get_detail(o, ' ')) for o in ques]
loop.run_until_complete(asyncio.wait(Tasks))
print(cnt1, cnt2)
f = open("./CSDN_精选/success.json", 'w')
json.dump(list(urls_success), f)
f.close()
