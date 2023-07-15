import scrapy
from so_spider.items import SoSpiderItem

cnt = 0

class StackoverflowSpider(scrapy.Spider):
    name = "stackoverflow"
    allowed_domains = ["stackoverflow.com"]
    links = []
    start_urls = []
    def __init__(self):    
        self.links = []
        with open ("../links/links.out", 'r') as f:
            for  line in f: self.links.append(str(line))
        self.start_urls = ["http://stackoverflow.com/" + self.links[0]]

    def parse(self, response):
        question = ''.join(response.xpath("//div[@id='question-header']/h1//text()").extract())
        question += ''.join(response.xpath("//div[@class='post-layout ']//div[@class='s-prose js-post-body']//text()").extract())
        answer = ''.join(response.xpath("//div[@id='answers']/div[2]//div[@class='s-prose js-post-body']//text()").extract())
        filename = "1.html"
        # open(filename, 'wb+').write(response.body)
        open(filename, 'w').write(str(len(self.links)))
        item = SoSpiderItem()
        item['question'] = question
        item['answer'] = answer
        global cnt
        cnt += 1
        yield item
        next_url = "http://stackoverflow.com/" + self.links[cnt]
        yield scrapy.Request(next_url)


