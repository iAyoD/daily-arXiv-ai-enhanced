import os

from daily_arxiv.arxiv_metadata import parse_categories, validate_paper_item


class DailyArxivPipeline:
    def __init__(self):
        self.target_categories = parse_categories(os.environ.get("CATEGORIES"))

    def process_item(self, item: dict, spider):
        validate_paper_item(item, self.target_categories)
        return item
