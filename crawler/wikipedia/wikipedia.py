#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, jsonlines
import asyncio
import aiohttp
import keywords
import time
from lxml import etree

ques = keywords.keywords

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

url_ = "https://zh.wikipedia.org/w/index.php?search={}"
with open("./wikipedia_{}/success.json".format(keywords.name), 'r') as f:
    urls_success = set(json.load(f))
cnt1 = 0 # all urls
cnt2 = 0 # urls succeeded
cnt3 = 0 # urls have no search result

async def write_QA(question, answer, knowledge_point):
    QA_ = {"Answer": answer, "Konwledge_Point": knowledge_point, "Question": question, "Tag": keywords.name}
    with jsonlines.open("./wikipedia_{}/{}_data.jsonl".format(keywords.name, knowledge_point.replace('/', 'or')), 'a') as writer:
                writer.write(QA_)

async def get_question(source, key, tree, title):
    str_ = source.format(key)
    text = key
    cnt = 0
    for i in range(7, 1, -1):
        if tree.xpath('count(' + str_ + '/parent::h{})'.format(i)):
            cnt = i
    for i in range(7, 2, -1):
        if cnt == i:
            fa_str = str_ + "/../preceding-sibling::h{}/span[@class='mw-headline']//text()"
            for j in range(1, i - 1):
                if tree.xpath("count(" + fa_str.format(i - j) + ")"):
                    pa = tree.xpath(fa_str.format(i - j))
                    pa = pa[-1]
                    text = pa + '的' + text
                    cnt = cnt - 1
                    break
    return str("什么是" + title + "的" + text)

def get_keys_intoc(tree):
    keys = []
    keys_str = ""
    content = ''
    if sum:
        content = tree.xpath("//div[@id='toc']/preceding-sibling::*[not(self::table) and not(self::div) and not(self::style)]//text()")
        keys = tree.xpath("//div[@id='toc']/ul//span[@class='toctext']//text()")
    else:
        content = tree.xpath("//div[@id='toc']/../preceding-sibling::*[not(self::table) and not(self::div) and not(self::style)]//text()")
        cnt = 0

        if int(tree.xpath("count(//div[@id='toc']/parent::div[@class='toclimit-2'])")): cnt = 2
        elif int(tree.xpath("count(//div[@id='toc']/parent::div[@class='toclimit-3'])")): cnt = 3
        elif int(tree.xpath("count(//div[@id='toc']/parent::div[@class='toclimit-4'])")): cnt = 4
        elif int(tree.xpath("count(//div[@id='toc']/parent::div[@class='toclimit-5'])")): cnt = 5
        keys_str = "//div[@id='toc']/ul/li/a/@href"
        s1 = " | //div[@id='toc']/ul/li{}/a/@href"
        ss = "/ul/li"
        for i in range(2, cnt):
            keys_str = keys_str + s1.format(ss)
            ss = ss + "/ul/li"
        keys = tree.xpath(keys_str)
        for i, key in enumerate(keys):
            keys[i] = key[1:]
    return keys, content

def get_keys(tree):
    content = ''
    keys = []
    keys = tree.xpath("//span[@class='mw-headline']/@id")
    content = tree.xpath("//span[@class='mw-headline'][1]/../preceding-sibling::*[not(self::table) and not(self::div) and not(self::style)]//text()")
    return keys, content

async def get_redirect(response):
    if str(response.url) in urls_success: return
    global cnt1, cnt2
    cnt1 += 1
    tree = etree.HTML(await response.text())
    title = tree.xpath("//h1[@id='firstHeading']//text()")[0]
    # sum = int(tree.xpath("count(//div[@id='toc']/parent::div[@class='mw-parser-output'])"))
    content = ''
    keys = []
    # print(tree.xpath("count(//div[@id='toc'])"))
    if tree.xpath("count(//div[@id='toc'])"):
        keys, content = get_keys_intoc(tree)
    else:
        keys, content = get_keys(tree)

    await write_QA("什么是" + title, ('').join(content), title)
    key_dic = {}
    source = "//span[@class='mw-headline' and @id='{}']"
    for _, key in enumerate(keys):
        if key not in key_dic:
            key_dic[key] = 1
        else:
            key_dic[key] += 1
            key += '_{}'.format(key_dic[key])
        
        question = await get_question(source, key, tree, title)
        str_ = source.format(key.replace(' ', '_'))
        cnt = 0
        for i in range(7, 1, -1):
            if tree.xpath('count(' + str_ + '/parent::h{})'.format(i)):
                cnt = i
        p1 = tree.xpath(str_ + '/..')
        p2 = tree.xpath(str_ + "/../following-sibling::h{}".format(cnt) + " | " + str_ + "/../following-sibling::h{}".format(cnt - 1) + " | " + str_ + "/../following-sibling::table")
        p1 = p1[0]
        parent1 = p1.getparent()
        start = parent1.index(p1)
        end = 0
        if len(p2):
            p2 = p2[0]
            end = parent1.index(p2)
        else:
            p2 = (tree.xpath(str_ + "/../following-sibling::*"))[-1]
            end = parent1.index(p2) + 1
        text = ""
        for j in range(start + 1, end):
            element = parent1[j]
            if element.tag == 'style': continue
            text_nodes = element.xpath('.//text()')
            text += ('').join(text_nodes)
        await write_QA(question, text, title)
    print("{} get!".format(title))
    cnt2 += 1
    urls_success.add(str(response.url))

async def get_detail(response):
    tree = etree.HTML(await response.text())
    node = tree.xpath("//ul[@class='mw-search-results']/li")
    if (len(node) == 0): return
    global cnt3
    node = node[0]
    new_url = "https://zh.wikipedia.org" + str((node.xpath(".//div[@class='mw-search-result-heading']/a/@href"))[0])
    cnt3 -= 1
    if new_url in urls_success: return 
    print(new_url)
    async with aiohttp.ClientSession() as session:
        async with session.get(new_url, proxy = "http://127.0.0.1:7890") as response:
            await get_redirect(response)

async def get_knowledge(o):
    url = url_.format(o)
    global cnt3
    async with aiohttp.ClientSession() as session:
        async with session.get(url, proxy = "http://127.0.0.1:7890") as response:
            if '?' in str(response.url): 
                cnt3 += 1
                await get_detail(response)
            else:
                await get_redirect(response)            

# Tasks = [asyncio.ensure_future(get_knowledge(ques[i])) for i in range(0, 20)]
Tasks = [asyncio.ensure_future(get_knowledge(q)) for q in ques]
# Tasks = [asyncio.ensure_future(get_knowledge("栈"))]
loop.run_until_complete(asyncio.wait(Tasks))
print(cnt1, cnt2, cnt3, len(list(urls_success)))
with open("./wikipedia_{}/success.json".format(keywords.name), 'w') as f:
    json.dump(list(urls_success), f)
