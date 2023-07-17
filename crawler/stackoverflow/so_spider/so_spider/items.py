# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SoSpiderItem(scrapy.Item):
    Question = scrapy.Field()
    Answer = scrapy.Field()
    Knowledge_point = scrapy.Field()
    Tag = scrapy.Field()
