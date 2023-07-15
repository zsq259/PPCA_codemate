#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup

# 设置要爬取的Wikipedia页面的URL
proxy = {
    'http': '127.0.0.1:7890',
    'https': '127.0.0.1:7890'
}
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}
url = "https://zh.wikipedia.org"
response = requests.get(url, headers = headers, proxies = proxy)
print(response.status_code)
