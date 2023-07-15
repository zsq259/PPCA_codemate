#!/usr/bin/env python3
#coding=utf-8
import re, json

def extract_code(text):
    # 使用正则表达式匹配代码部分
    # print(text)
    code_pattern = r'```[A-Za-z]*\n([\s\S]+?)\n```'
    # code_pattern = r'```[A-Za-z]*```'
    code_matches = re.findall(code_pattern, text)
    print(code_matches)
    # 返回匹配到的代码部分
    return code_matches

# 示例文本
text = '''
这是一段普通的文本。

这是一段包含代码的文本：

```python
def hello_world():
    print("Hello, World!")

hello_world()
```

再来一段代码

java```
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

结束

'''

code_snippets = extract_code(text)
for code in code_snippets:
    print(code)

# file_name = "test.jsonl"
# with open(file_name, 'r', encoding = 'utf-8') as f:
#     for line in f:
#         record = json.loads(line)
#         code1 = extract_code(record["Answer"])
#         code2 = extract_code(record["Question"])
#         print(code1, code2)

