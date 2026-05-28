import json
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
