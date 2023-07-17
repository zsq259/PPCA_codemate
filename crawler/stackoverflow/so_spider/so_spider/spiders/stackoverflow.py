import scrapy, jsonlines
from so_spider.items import SoSpiderItem

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
        answer = ''
        children = response.xpath("//div[@class='post-layout ']//div[@class='s-prose js-post-body']/*")
        for child in children:
            if float(child.xpath("count(./div)").extract()[0]) > 0: continue
            if float(child.xpath("count(.//code)").extract()[0]) > 0:
                question += "```\n" + ''.join(child.xpath(".//text()").extract()) + "```\n"
            else:
                question += ''.join(child.xpath(".//text()").extract())
        
        children = response.xpath("//div[@id='answers']/div[2]//div[@class='s-prose js-post-body']/*")
        for child in children:
            if float(child.xpath("count(./div)").extract()[0]) > 0: continue
            if float(child.xpath("count(.//code)").extract()[0]) > 0:
                answer += "```\n" + ''.join(child.xpath(".//text()").extract()) + "```\n"
            else:
                answer += ''.join(child.xpath(".//text()").extract())
        
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

