#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#get about keywords CSDN with asynico

import asyncio, time, html, json, jsonlines, re
from lxml import etree
from bs4 import BeautifulSoup


from playwright.async_api import Playwright, async_playwright, expect

urls_success = set()
with open("./CSDN_精选/success.out", 'r') as f:
    for line in f: urls_success.add(str(line))

cnt1 = 0
cnt2 = 0

async def traverse(s):
    s = html.unescape(s)
    soup = BeautifulSoup(s, 'html.parser')
    text = soup.get_text()
    return text


async def run(playwright: Playwright, detail_url) -> None:
    global cnt1, cnt2
    cnt1 += 1
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(detail_url)
    time.sleep(1)
    html = await page.content()
    html = re.sub(r'<code(\s.*?)>', r'<code\1>```\n', html)
    html = re.sub(r'</code>', r'```\n</code>', html)
    tree = etree.HTML(html)
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
    with open("./CSDN_精选/success.out", 'a') as f:
        f.write(detail_url)
    with jsonlines.open("./CSDN_精选/data.jsonl", 'a') as writer:
        writer.write(QA_)
    await page.close()
    # ---------------------
    await context.close()
    await browser.close()


async def work(semaphore, detail_url) -> None:
    async with semaphore:
        if detail_url in urls_success: return
        async with async_playwright() as playwright:
            await run(playwright, detail_url)

ques = []
with open("links.out", 'r') as f:
    for line in f:
        ques.append(str(line))

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

semaphore = asyncio.Semaphore(5)
Tasks = [asyncio.ensure_future(work(semaphore, o)) for o in ques]
# Tasks = [asyncio.ensure_future(work(ques[i])) for i in range(70, 120)]
loop.run_until_complete(asyncio.wait(Tasks))
print(cnt1, cnt2)
