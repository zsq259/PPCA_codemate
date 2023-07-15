#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright, o) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://stackoverflow.com/")
    page.get_by_placeholder("Search…").click()
    page.get_by_placeholder("Search…").fill(o)
    page.get_by_placeholder("Search…").press("Enter")
    show = page.selector("xpath=//a[@title='Show 50 items per page']")
    print(show)
    if show != None: show.click()


    page.close()

    # ---------------------
    context.close()
    browser.close()


def work(o):
    with sync_playwright() as playwright:
        run(playwright)

work("primary")