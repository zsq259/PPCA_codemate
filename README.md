# PPCA_codemate

本项目分为两部分：爬虫部分实现了对一些网站数据的爬取；分类器部分实现了对爬取数据质量筛选的分类器的训练。

## 爬虫

爬虫学习&编写记录：[https://hastin-blog.cn/post/python%E7%88%AC%E8%99%AB%E5%AE%9E%E5%BD%95/](https://hastin-blog.cn/post/python%E7%88%AC%E8%99%AB%E5%AE%9E%E5%BD%95/)

本项目实现了对 CSDN 问答，Wikipedia，stackoverflow 三个网站的数据爬取。

所有爬虫均实现了成功爬取链接的记录，存储在对应的 `success.out` 中，在爬取时只爬取未成功的链接。即使程序运行中断，再次启动程序也能实现无重复无遗漏的爬取。

CSDN 精华 和 stackoverflow 的爬虫实现了对代码块的识别，爬取的代码块会用 ` ```\n ` 包裹起来。

爬取的内容为 `算法分析` 相关，搜索的关键词列表为：

```
"书籍和算法",
"Fibonacci数列",
"大O表示法",
"算法与数字",
"基本算术",
"模运算",
"素性测试",
"密码学",
"通用散列",
"分治算法",
"乘法",
"递归关系",
"归并排序",
"中位数",
"矩阵乘法",
"快速傅里叶变换",
"图的分解",
"图的深度优先搜索",
"图的强连通分量",
"图中的路径",
"广度优先搜索",
"边的长度",
"Dijkstra算法",
"优先队列实现",
"负权边的最短路径",
"有向无环图的最短路径",
"贪婪算法",
"最小生成树",
"哈夫曼编码",
"Horn公式",
"集合覆盖",
"动态规划",
"有向无环图的最短路径（重温）",
"最长递增子序列",
"编辑距离",
"背包问题",
"矩阵链乘法",
"最短路径",
"树中的独立集",
"线性规划和规约",
"线性规划简介",
"网络中的流",
"二分匹配",
"对偶性",
"零和博弈",
"单纯形算法",
"电路评估",
"NP完全问题",
"搜索问题",
"规约",
"应对NP完全问题",
"智能穷举搜索",
"近似算法"
```

以下是各网站的爬虫介绍。

### CSDN

CSDN 网站分为两部分：
- 搜索关键字后爬取对应的数据
- 在[CSDN问答的精华版块](https://ask.csdn.net/channel/1005?rewardType&stateType=0&sortBy=1&quick=6&essenceType=1&tagName=essence)爬取所有数据。

#### 第一部分

为保证数据形式和质量，在 CSDN 首页点击问答并勾选已采纳，再爬取获得的数据。

相关代码在 [`CSDN_1.py`](https://github.com/zsq259/PPCA_codemate/blob/main/crawler/CSDN/CSDN_1.py) 中。

中通过 `asyncio` 库，利用协程实现了并行爬虫。并且通过 `BeautifulSoup` 去除了 html 代码。

由于没有用模拟浏览器的方式，需要在搜索关键词后的页面爬取答案，再到问题详情页爬取问题（问题详情页的答案是动态渲染的）。

#### 第二部分

首先，需要进入 [https://ask.csdn.net/channel/1005?rewardType&stateType=0&sortBy=1&quick=6&essenceType=1&tagName=essence](https://ask.csdn.net/channel/1005?rewardType&stateType=0&sortBy=1&quick=6&essenceType=1&tagName=essence) 页面获得所有具体问答的链接。这一步骤在 [`CSDN.py`](https://github.com/zsq259/PPCA_codemate/blob/main/crawler/CSDN/CSDN.py) 中解决。代码主要利用 `playwright` ，模拟了鼠标滚轮下滑来获取所有链接。

获取完所有链接后，再进入到具体的问答页面获得数据。根据第一部分的经验，已知答案是动态渲染的，所以采取了利用 `playwright` 模拟浏览器的方式爬取。在此基础上，[`CSDN_2.py`](https://github.com/zsq259/PPCA_codemate/blob/main/crawler/CSDN/CSDN_2.py) 利用 `asyncio` 实现了并行爬虫，但由于同时爬取的链接太多会导致崩溃。而 [`CSDN_3.py`] 利用 `threading` 采取多线程爬虫，从而可以限制线程数量，但也需注意变量锁的设置。

此部分有一处细节在于，模拟浏览器进入页面后，页面并非立刻加载完成，所以需要使用 `time.sleep()` 等待一段时间，否则会无法获取信息。

### Wikipedia

具体的过程和需要注意的细节已经写在了 [blog 中](https://hastin-blog.cn/post/python%E7%88%AC%E8%99%AB%E5%AE%9E%E5%BD%95/)。

在 [`wikipedia.py`](https://github.com/zsq259/PPCA_codemate/blob/main/crawler/wikipedia/wikipedia.py) 中，实现了：

- 在搜索框中搜索关键词。

- 若不能直接跳转具体词条页面，则选取搜索结果的第一条跳转。

- 若有目录，则按照目录显示的各级标题作为所有问题（有些标题并未显示在目录中）；否则直接将所有标题作为问题。

- 将所有问题（各级标题）下对应的文字进行爬取，并拼接为答案组成问答对。

- 每个问题将会爬取当前标题与下一个同级的标题或页面结尾之间的所有文字。

### stackoverflow

与 CSDN 没有太大区别，但是有人机验证和请求太多封 ip 的反爬机制。

人工进行人机验证，在接下来的 5min 内网站不会跳转人机验证页面，在这段时间利用爬虫爬取。

* [ ] 需要建立 ip 池解决请求太多被封 ip 的问题。

## 分类器

学习记录：[https://hastin-blog.cn/post/%E6%96%87%E6%9C%AC%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E5%92%8C%E5%88%86%E7%B1%BB%E5%99%A8%E7%9A%84%E8%AE%AD%E7%BB%83/](https://hastin-blog.cn/post/%E6%96%87%E6%9C%AC%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E5%92%8C%E5%88%86%E7%B1%BB%E5%99%A8%E7%9A%84%E8%AE%AD%E7%BB%83/)

使用 `jieba` 进行分词。

利用 `sklearn`， 使用 `TfidfVectorizer` 实现向量化，再使用 `RandomForestClassifier` 进行随机森林算法。

