#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import html2markdown
from bs4 import BeautifulSoup
import html, json, jsonlines

def convert_html_file_to_markdown(file_path, output_file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        for s in file:
            s = html.unescape(s)
            soup = BeautifulSoup(s, 'html.parser')
            text = soup.get_text()
            record = json.load(text)
            with jsonlines.open(output_file_path, 'a') as writer:
                writer.write(text)

input_file_path = "./CSDN/CSDN_程序设计/变量定义_data.jsonl"
output_file_path = "./CSDN/CSDN_程序设计/变量定义.jsonl"
convert_html_file_to_markdown(input_file_path, output_file_path)
