import scrapy, jsonlines, re
from so_spider.items import SoSpiderItem
from scrapy.selector import Selector

class StackoverflowSpider(scrapy.Spider):
    name = "stackoverflow"
    allowed_domains = ["stackoverflow.com"]
    start_urls = []
    def __init__(self, knowledge_point = None):    
        self.urls_secceeded = set([])
        with open("success.out", 'r') as f:
            for line in f:
                self.urls_secceeded.add(str(line))
        cnt = 0
        self.knowledge_point = knowledge_point
        print(self.knowledge_point)
        with open ("../links/{}_links.out".format(self.knowledge_point), 'r') as f:
            for  line in f:
                cnt += 1
                url = "https://stackoverflow.com" + str(line)
                if url in self.urls_secceeded: continue
                self.start_urls.append(url)
        print(cnt)
        

    def parse(self, response):
        question = ''.join(response.xpath("//div[@id='question-header']/h1//text()").extract())
        html = response.body.decode("utf-8")
        html = re.sub(r'<code>', r'<code>```\n', html)
        html = re.sub(r'</code>', r'```\n</code>', html)
        selector = Selector(text=html)
        question += ''.join(selector.xpath("//div[@class='post-layout ']//div[@class='s-prose js-post-body']//text()").extract())
        answer = ''.join(selector.xpath("//div[@id='answers']/div[2]//div[@class='s-prose js-post-body']//text()").extract())
        item = SoSpiderItem()
        item['Question'] = question
        item['Answer'] = answer
        item['Knowledge_point'] = str(self.knowledge_point)
        item['Tag'] = '算法分析'
        self.urls_secceeded.add(response.url)
        with open("success.out", 'a') as f:
            f.write(str(response.url) + '\n')
        with jsonlines.open("datas/{}_data.jsonl".format(self.knowledge_point), 'a') as writer:
            writer.write(dict(item))
        yield item

