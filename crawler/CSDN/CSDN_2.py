#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio, time, html, json, jsonlines
from lxml import etree
from bs4 import BeautifulSoup


from playwright.async_api import Playwright, async_playwright, expect

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


async def run(playwright: Playwright, detail_url) -> None:
    
    global cnt1, cnt2
    cnt1 += 1
    # browser = await playwright.chromium.launch(headless=False)
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(detail_url)
    time.sleep(1)
    tree = etree.HTML(await page.content())
    title = tree.xpath("//section[@class='title-box']/h1/text()")[0]
    question = tree.xpath('//section[@class="question_show_box"]//div[@class="md_content_show"]//text()')
    answer = tree.xpath("(//div[@class='answer_box']//div[@class='answer-content-item'])[1]//text()")
    answer = '\n'.join(answer)
    answer = await traverse(answer)
    title = await traverse(title)
    question = ('\n').join(question)
    question = ('\n').join([title, question])
    question = await traverse(question)
    QA_ = {"Answer": answer, "Konwledge_Point": '', "Question": question, "Tag": ''}
    urls_success.add(detail_url)
    cnt2 += 1
    print('#', end='')
    # print(detail_url)
    # print(keywords.name , o)
    with jsonlines.open("./CSDN_精选/data.jsonl", 'a') as writer:
        writer.write(QA_)

    await page.close()
    # ---------------------
    await context.close()
    await browser.close()


async def work(detail_url) -> None:
    if detail_url in urls_success: return
    async with async_playwright() as playwright:
        await run(playwright, detail_url)


# asyncio.run(work("https://ask.csdn.net/questions/7976191"))

ques = []
with open("links.out", 'r') as f:
    for line in f:
        ques.append(str(line))

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


# Tasks = [asyncio.ensure_future(work(o)) for o in ques]
Tasks = [asyncio.ensure_future(work(ques[i])) for i in range(70, 120)]
loop.run_until_complete(asyncio.wait(Tasks))
print(cnt1, cnt2)
f = open("./CSDN_精选/success.json", 'w')
json.dump(list(urls_success), f)
f.close()
