import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
import enhance


class FakeChain:
    def invoke(self, _payload):
        content = json.dumps({
            "tldr": "short",
            "motivation": "why",
            "method": "how",
            "result": "result",
            "conclusion": "done",
        })
        return type("Response", (), {"content": content})()


class FakeGitHubResponse:
    status_code = 200

    def json(self):
        return {
            "stargazers_count": 42,
            "pushed_at": "2026-05-27T12:00:00Z",
        }


class EnhanceTests(unittest.TestCase):
    def test_process_single_item_generates_ai_without_sensitive_check(self):
        item = {"id": "paper-id", "summary": "plain abstract"}

        result = enhance.process_single_item(
            FakeChain(),
            FakeChain(),
            item,
            "Chinese",
            1,
        )

        self.assertEqual(result["id"], "paper-id")
        self.assertEqual(result["AI"]["tldr"], "short")

    def test_process_single_item_detects_github_from_external_urls(self):
        item = {
            "id": "paper-id",
            "summary": "plain abstract",
            "external_urls": ["https://github.com/example/robot-code"],
        }

        with patch.object(enhance.requests, "get", return_value=FakeGitHubResponse()):
            result = enhance.process_single_item(
                FakeChain(),
                FakeChain(),
                item,
                "Chinese",
                1,
            )

        self.assertEqual(result["code_url"], "https://github.com/example/robot-code")
        self.assertEqual(result["code_stars"], 42)
        self.assertEqual(result["code_last_update"], "2026-05-27")

    def test_process_single_item_detects_github_io_from_external_urls(self):
        item = {
            "id": "paper-id",
            "summary": "plain abstract",
            "external_urls": ["https://example.github.io/robot-project/"],
        }

        result = enhance.process_single_item(
            FakeChain(),
            FakeChain(),
            item,
            "Chinese",
            1,
        )

        self.assertEqual(result["code_url"], "https://example.github.io/robot-project/")


if __name__ == "__main__":
    unittest.main()
