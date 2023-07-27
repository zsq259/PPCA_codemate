#!/usr/bin/env python3
#coding=utf-8
import openai

# 设置你的OpenAI API密钥
api_key = "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"
openai.api_key = api_key

def synonym_replacement(prompt):
    print("ok")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        stop=["\n"]
    )
    return response.choices[0].text.strip()

# 原始句子
original_sentence = "这是一个很有趣的问题。"

print("原始句子:", original_sentence)
# 使用ChatGPT增强数据
augmented_sentence = synonym_replacement(original_sentence)
print("增强后的句子:", augmented_sentence)
