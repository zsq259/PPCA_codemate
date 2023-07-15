#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from playwright.sync_api import Playwright, sync_playwright, expect
from keywords import keywords
import time, threading

def run(playwright: Playwright, o) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://stackoverflow.com/")
    page.get_by_placeholder("Search…").click()
    page.get_by_placeholder("Search…").fill(o)
    page.get_by_placeholder("Search…").press("Enter")
    time.sleep(0.2)
    show = page.query_selector("xpath=//a[@title='Show 50 items per page']")
    if show != None: show.click()
    cnt = 0
    while (True):
        time.sleep(1)
        elements = page.query_selector_all("xpath=//h3[@class='s-post-summary--content-title']/a")
        for element in elements:
            with open("links/{}.out".format(o), 'a') as f:
                f.write(element.get_attribute('href') + '\n')
        cnt += 1
        next = page.query_selector("xpath=//a[@rel='next']")
        if next == None: break
        next.click()
    print(o, "total page:", cnt, "get!")

    page.close()

    # ---------------------
    context.close()
    browser.close()


def work(o):
    with sync_playwright() as playwright:
        run(playwright, o)

for o in keywords: work(o)