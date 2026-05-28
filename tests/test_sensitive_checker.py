import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
import enhance


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self.payload = payload if payload is not None else {}
        self.headers = headers if headers is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise enhance.requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        return self.payload


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


class FakeSensitiveChecker:
    def __init__(self):
        self.calls = []

    def is_sensitive(self, content, item_id, field_name):
        self.calls.append((content, item_id, field_name))
        return False


class SensitiveCheckerTests(unittest.TestCase):
    def test_retries_after_rate_limit(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "0"}),
            FakeResponse(200, payload={"sensitive": False}),
        ]
        calls = []

        def fake_post(url, json, timeout):
            calls.append((url, json, timeout))
            return responses.pop(0)

        checker = enhance.SensitiveChecker(
            min_interval_seconds=0,
            max_attempts=2,
            retry_base_seconds=0,
        )

        with (
            patch.object(enhance.requests, "post", side_effect=fake_post),
            patch.object(enhance.time, "sleep"),
        ):
            self.assertFalse(checker.is_sensitive("hello", "paper-id", "summary"))

        self.assertEqual(len(calls), 2)

    def test_process_single_item_checks_ai_fields_in_one_request(self):
        checker = FakeSensitiveChecker()
        item = {"id": "paper-id", "summary": "plain abstract"}

        result = enhance.process_single_item(
            FakeChain(),
            FakeChain(),
            item,
            "Chinese",
            1,
            checker,
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            [field_name for _content, _item_id, field_name in checker.calls],
            ["summary", "AI"],
        )


if __name__ == "__main__":
    unittest.main()
