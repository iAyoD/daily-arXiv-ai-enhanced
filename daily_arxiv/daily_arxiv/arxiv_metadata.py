import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

DEFAULT_LIST_SHOW = 2000
REQUIRED_PAPER_FIELDS = ("id", "pdf", "abs", "authors", "title", "categories", "summary")


@dataclass(frozen=True)
class ListedPaper:
    id: str
    categories: List[str]


def parse_categories(raw_categories: Optional[str]) -> List[str]:
    categories_text = raw_categories if raw_categories is not None else "cs.CV"
    categories = [category.strip() for category in categories_text.split(",")]
    categories = [category for category in categories if category]
    if not categories:
        raise ValueError("CATEGORIES must contain at least one arXiv category")
    return categories


def read_positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def listing_url(category: str, show: int) -> str:
    return f"https://arxiv.org/list/{category}/new?skip=0&show={show}"


def extract_listed_papers(selector, target_categories: Sequence[str]) -> List[ListedPaper]:
    target_category_set = set(target_categories)
    if not target_category_set:
        raise ValueError("target_categories must not be empty")

    replacement_start = _replacement_start_index(selector)
    listed_papers = []

    for paper in selector.css("dl dt"):
        paper_anchor = paper.css("a[name^='item']::attr(name)").get()
        if not paper_anchor:
            continue

        item_index = _parse_item_index(paper_anchor)
        if replacement_start is not None and item_index >= replacement_start:
            continue

        abstract_link = paper.css("a[title='Abstract']::attr(href)").get()
        if not abstract_link:
            raise ValueError(f"Missing abstract link for list item {paper_anchor}")

        arxiv_id = normalize_arxiv_id(abstract_link.split("/")[-1])
        paper_dd = paper.xpath("following-sibling::dd[1]")
        if not paper_dd:
            raise ValueError(f"Missing metadata block for arXiv paper {arxiv_id}")

        categories = _extract_categories(paper_dd)
        if not categories:
            raise ValueError(f"Missing category metadata for arXiv paper {arxiv_id}")

        if target_category_set.intersection(categories):
            listed_papers.append(ListedPaper(id=arxiv_id, categories=categories))

    return listed_papers


def parse_abs_page(selector, paper_id: str, target_categories: Sequence[str]) -> Dict:
    normalized_id = normalize_arxiv_id(paper_id)
    item = {
        "id": normalized_id,
        "pdf": f"https://arxiv.org/pdf/{normalized_id}",
        "abs": f"https://arxiv.org/abs/{normalized_id}",
        "authors": _extract_authors(selector),
        "title": _clean_descriptor_text(selector.css("h1.title ::text, h1.title::text").getall(), "Title:"),
        "categories": _extract_abs_categories(selector),
        "comment": _extract_optional_descriptor_text(
            selector.css("td.tablecell.comments ::text, td.tablecell.comments::text").getall()
        ),
        "summary": _clean_descriptor_text(
            selector.css("blockquote.abstract ::text, blockquote.abstract::text").getall(),
            "Abstract:",
        ),
    }
    validate_paper_item(item, target_categories)
    return item


def validate_paper_item(item: Dict, target_categories: Optional[Iterable[str]] = None) -> None:
    missing_fields = [field for field in REQUIRED_PAPER_FIELDS if field not in item]
    if missing_fields:
        raise ValueError(
            f"Paper {item.get('id', 'unknown')} missing required fields: "
            + ", ".join(missing_fields)
        )

    if not isinstance(item["authors"], list) or not item["authors"]:
        raise ValueError(f"Paper {item['id']} must have a non-empty authors list")
    if not isinstance(item["categories"], list) or not item["categories"]:
        raise ValueError(f"Paper {item['id']} must have a non-empty categories list")
    for text_field in ("id", "pdf", "abs", "title", "summary"):
        if not isinstance(item[text_field], str) or not item[text_field].strip():
            raise ValueError(f"Paper {item.get('id', 'unknown')} has empty field: {text_field}")

    if target_categories is not None and not set(target_categories).intersection(item["categories"]):
        raise ValueError(f"Paper {item['id']} does not match target categories")


def _replacement_start_index(selector) -> Optional[int]:
    for list_item in selector.css("div[id=dlpage] ul li"):
        label = " ".join(text.strip() for text in list_item.css("::text").getall())
        href = list_item.css("a::attr(href)").get()
        if href and "Replacements" in label:
            return _parse_item_index(href.split("#")[-1])
    return None


def _parse_item_index(anchor: str) -> int:
    match = re.fullmatch(r"item(\d+)", anchor)
    if not match:
        raise ValueError(f"Unexpected arXiv list item anchor: {anchor}")
    return int(match.group(1))


def _extract_categories(paper_dd) -> List[str]:
    subjects_text = " ".join(
        text.strip()
        for text in paper_dd.css(".list-subjects ::text, .list-subjects::text").getall()
        if text.strip()
    )
    categories = re.findall(r"\(([^)]+)\)", subjects_text)
    return _unique_preserving_order(category.strip() for category in categories if category.strip())


def _extract_authors(selector) -> List[str]:
    authors = [
        author.strip()
        for author in selector.css("div.authors a::text").getall()
        if author.strip()
    ]
    return authors


def _extract_abs_categories(selector) -> List[str]:
    subjects_text = " ".join(
        text.strip()
        for text in selector.css("td.tablecell.subjects ::text, td.tablecell.subjects::text").getall()
        if text.strip()
    )
    categories = re.findall(r"\(([^)]+)\)", subjects_text)
    return _unique_preserving_order(category.strip() for category in categories if category.strip())


def _clean_descriptor_text(text_parts: Iterable[str], descriptor: str) -> str:
    text = " ".join(text_part.strip() for text_part in text_parts if text_part.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith(descriptor):
        text = text[len(descriptor):].strip()
    return text


def _extract_optional_descriptor_text(text_parts: Iterable[str]) -> Optional[str]:
    text = _clean_descriptor_text(text_parts, "")
    return text or None


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values
