#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import requests
from parsel import Selector

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from daily_arxiv.arxiv_metadata import (
    DEFAULT_LIST_SHOW,
    extract_listed_papers,
    listing_url,
    parse_categories,
    read_positive_int_env,
    validate_paper_item,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Crawled JSONL data file")
    return parser.parse_args()


def load_crawled_papers(data_file: Path):
    if not data_file.exists():
        raise FileNotFoundError(f"Crawl output file does not exist: {data_file}")

    papers = []
    with data_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {data_file}") from e
    return papers


def expected_ids_from_arxiv(categories, show):
    expected_ids = []
    seen_ids = set()
    for category in categories:
        url = listing_url(category, show)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        selector = Selector(text=resp.text)
        for paper in extract_listed_papers(selector, categories):
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                expected_ids.append(paper.id)
    return expected_ids


def validate_crawl(data_file: Path):
    categories = parse_categories(os.environ.get("CATEGORIES"))
    show = read_positive_int_env("ARXIV_LIST_SHOW", DEFAULT_LIST_SHOW)
    papers = load_crawled_papers(data_file)

    ids = []
    seen_ids = set()
    duplicate_ids = set()
    for paper in papers:
        validate_paper_item(paper, categories)
        paper_id = paper["id"]
        if paper_id in seen_ids:
            duplicate_ids.add(paper_id)
        seen_ids.add(paper_id)
        ids.append(paper_id)

    if duplicate_ids:
        raise RuntimeError("Crawl output contains duplicate paper IDs: " + ", ".join(sorted(duplicate_ids)))

    expected_ids = expected_ids_from_arxiv(categories, show)
    expected_id_set = set(expected_ids)
    crawled_id_set = set(ids)

    missing_ids = [paper_id for paper_id in expected_ids if paper_id not in crawled_id_set]
    extra_ids = [paper_id for paper_id in ids if paper_id not in expected_id_set]

    if missing_ids:
        raise RuntimeError("Crawl output is missing expected papers: " + ", ".join(missing_ids))
    if extra_ids:
        raise RuntimeError("Crawl output contains unexpected papers: " + ", ".join(extra_ids))

    print(
        f"Validated crawl output: {len(ids)} papers for categories {', '.join(categories)}",
        file=sys.stderr,
    )


def main():
    args = parse_args()
    validate_crawl(Path(args.data))


if __name__ == "__main__":
    main()
