import scrapy
import os

from daily_arxiv.arxiv_metadata import (
    DEFAULT_LIST_SHOW,
    extract_listed_papers,
    listing_url,
    parse_abs_page,
    parse_categories,
    read_positive_int_env,
)


class ArxivSpider(scrapy.Spider):
    name = "arxiv"  # 爬虫名称
    allowed_domains = ["arxiv.org"]  # 允许爬取的域名

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_categories = parse_categories(os.environ.get("CATEGORIES"))
        self.list_show = read_positive_int_env("ARXIV_LIST_SHOW", DEFAULT_LIST_SHOW)
        self.seen_ids = set()
        self.start_urls = [
            listing_url(category, self.list_show) for category in self.target_categories
        ]  # 起始URL（计算机科学领域的最新论文）

    def parse(self, response):
        listed_papers = extract_listed_papers(response, self.target_categories)
        new_ids = []
        for paper in listed_papers:
            if paper.id in self.seen_ids:
                continue
            self.seen_ids.add(paper.id)
            new_ids.append(paper.id)
            self.logger.info("Found paper %s with categories %s", paper.id, set(paper.categories))

        for paper_id in new_ids:
            yield response.follow(
                f"/abs/{paper_id}",
                callback=self.parse_abs,
                meta={"paper_id": paper_id},
            )

    def parse_abs(self, response):
        paper_id = response.meta["paper_id"]
        yield parse_abs_page(response, paper_id, self.target_categories)
