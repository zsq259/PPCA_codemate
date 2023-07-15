# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SoSpiderItem(scrapy.Item):
    question = scrapy.Field()
    answer = scrapy.Field()
