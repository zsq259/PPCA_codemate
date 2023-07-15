import scrapy


class StackoverflowSpider(scrapy.Spider):
    name = "stackoverflow"
    allowed_domains = ["stackoverflow.com"]
    start_urls = ["http://stackoverflow.com/"]

    def parse(self, response):
        pass
