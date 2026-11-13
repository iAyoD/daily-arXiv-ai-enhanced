import sys
import unittest
from pathlib import Path

from parsel import Selector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daily_arxiv"))
from daily_arxiv.arxiv_metadata import (
    extract_listed_papers,
    parse_abs_page,
    parse_categories,
    validate_paper_item,
)


LISTING_HTML = """
<html>
  <body>
    <div id="dlpage">
      <ul>
        <li><a href="#item0">New submissions</a></li>
        <li><a href="#item3">Cross-lists</a></li>
        <li><a href="#item5">Replacements</a></li>
      </ul>
      <dl>
        <dt><a name="item1"></a><a title="Abstract" href="/abs/2605.00001">abs</a></dt>
        <dd><div class="list-subjects">Subjects:
          <span class="primary-subject">Robotics (cs.RO)</span>
        </div></dd>
        <dt><a name="item2"></a><a title="Abstract" href="/abs/2605.00002v2">abs</a></dt>
        <dd><div class="list-subjects">Subjects:
          Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV)
        </div></dd>
        <dt><a name="item3"></a><a title="Abstract" href="/abs/2605.00003">abs</a></dt>
        <dd><div class="list-subjects">Subjects:
          Machine Learning (cs.LG); Robotics (cs.RO)
        </div></dd>
        <dt><a name="item5"></a><a title="Abstract" href="/abs/2605.00004">abs</a></dt>
        <dd><div class="list-subjects">Subjects:
          Robotics (cs.RO)
        </div></dd>
      </dl>
    </div>
  </body>
</html>
"""

ABS_HTML = """
<html>
  <body>
    <h1 class="title"><span class="descriptor">Title:</span> Example Robot Paper</h1>
    <div class="authors">
      <span class="descriptor">Authors:</span>
      <a>First Author</a>, <a>Second Author</a>
    </div>
    <blockquote class="abstract mathjax">
      <span class="descriptor">Abstract:</span>
      This paper studies robots. See <a href="https://github.com/example/robot-code">this https URL</a>.
    </blockquote>
    <table>
      <tr>
        <td class="tablecell comments mathjax">10 pages</td>
      </tr>
      <tr>
        <td class="tablecell subjects">
          Robotics (cs.RO); Machine Learning (cs.LG)
        </td>
      </tr>
    </table>
    <a href="https://arxiv.org/pdf/2605.00001">PDF</a>
    <a href="https://example.github.io/robot-project/">Project</a>
  </body>
</html>
"""

RECENT_LISTING_HTML = """
<div id="dlpage">
  <dl>
    <h3>Fri, 13 Nov 2026 (showing 1 of 1 entries )</h3>
    <dt><a name="item1"></a><a title="Abstract" href="/abs/2611.17330">abs</a></dt>
    <dd><div class="list-subjects">Robotics (cs.RO)</div></dd>
  </dl>
  <dl>
    <h3>Wed, 11 Nov 2026 (showing 1 of 1 entries )</h3>
    <dt><a name="item2"></a><a title="Abstract" href="/abs/2611.12500">abs</a></dt>
    <dd><div class="list-subjects">Robotics (cs.RO)</div></dd>
  </dl>
</div>
"""


class ArxivMetadataTests(unittest.TestCase):
    def test_backfill_selects_only_the_requested_publication_date(self):
        papers = extract_listed_papers(
            Selector(text=RECENT_LISTING_HTML), ["cs.RO"], "2026-11-13"
        )
        self.assertEqual([paper.id for paper in papers], ["2611.17330"])

    def test_backfill_returns_no_papers_on_a_day_without_announcements(self):
        self.assertEqual(
            extract_listed_papers(
                Selector(text=RECENT_LISTING_HTML), ["cs.RO"], "2026-11-12"
            ),
            [],
        )

    def test_backfill_rejects_dates_outside_the_available_list(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            extract_listed_papers(
                Selector(text=RECENT_LISTING_HTML), ["cs.RO"], "2026-11-10"
            )

    def test_backfill_rejects_a_truncated_listing(self):
        partial_html = RECENT_LISTING_HTML.replace("showing 1 of 1", "showing 1 of 2")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            extract_listed_papers(Selector(text=partial_html), ["cs.RO"], "2026-11-13")

    def test_extract_listed_papers_filters_replacements_and_normalizes_versions(self):
        papers = extract_listed_papers(Selector(text=LISTING_HTML), ["cs.RO", "cs.CV"])

        self.assertEqual([paper.id for paper in papers], ["2605.00001", "2605.00002", "2605.00003"])
        self.assertEqual(papers[1].categories, ["cs.AI", "cs.CV"])

    def test_parse_categories_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            parse_categories(" , ")

    def test_validate_paper_item_requires_target_category(self):
        item = {
            "id": "2605.00001",
            "pdf": "https://arxiv.org/pdf/2605.00001",
            "abs": "https://arxiv.org/abs/2605.00001",
            "authors": ["Author"],
            "title": "Title",
            "categories": ["cs.RO"],
            "summary": "Abstract",
        }

        validate_paper_item(item, ["cs.RO"])
        with self.assertRaises(ValueError):
            validate_paper_item(item, ["cs.CV"])

    def test_parse_abs_page_extracts_required_metadata(self):
        item = parse_abs_page(Selector(text=ABS_HTML), "2605.00001v2", ["cs.RO"])

        self.assertEqual(item["id"], "2605.00001")
        self.assertEqual(item["title"], "Example Robot Paper")
        self.assertEqual(item["authors"], ["First Author", "Second Author"])
        self.assertEqual(item["summary"], "This paper studies robots. See this https URL .")
        self.assertEqual(item["categories"], ["cs.RO", "cs.LG"])
        self.assertEqual(item["comment"], "10 pages")
        self.assertEqual(
            item["external_urls"],
            [
                "https://github.com/example/robot-code",
                "https://example.github.io/robot-project/",
            ],
        )


if __name__ == "__main__":
    unittest.main()
