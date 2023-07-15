#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests

url = "https://leetcode.com/problemset/all/"

# 发送GET请求获取页面内容
response = requests.get(url)

# 检查响应状态码，判断请求是否成功
if response.status_code == 200:
    # 打印页面内容
    print(response.text)
else:
    # 请求失败
    print("请求失败，状态码：", response.status_code)