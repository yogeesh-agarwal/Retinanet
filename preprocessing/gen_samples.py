import os
import random
import pickle
from icrawler.builtin import BingImageCrawler

def download_images(query , num_images , download_dir):
    bing_crawler = BingImageCrawler(downloader_threads=4 , storage={'root_dir': download_dir})
    bing_crawler.crawl(keyword=query , filters=None , offset=0 , max_num=num_images)

if __name__ == "__main__":
    num_images = 500
    neg_search_queries = ["building" , "human" , "dog" , "cat"]
    pos_search_queries = ["car"]
    download_dir = "/home/yogeesh/yogeesh/datasets/car/data/extra_data/testing/"
    for query in pos_search_queries:
        curr_download_dir = os.path.join(download_dir , query)
        if os.path.exists(curr_download_dir):
            os.makedirs(curr_download_dir)

        download_images(query , num_images , curr_download_dir)
