import json
import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

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
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {
            "stargazers_count": 42,
            "pushed_at": "2026-05-27T12:00:00Z",
        }

    def json(self):
        return self.payload


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

    def test_github_metadata_retries_timeout_then_succeeds(self):
        item = {
            "id": "paper-id",
            "summary": "plain abstract",
            "external_urls": ["https://github.com/example/robot-code"],
        }

        with (
            patch.dict(enhance.os.environ, {"TOKEN_GITHUB": "test-token"}),
            patch.object(
                enhance.requests,
                "get",
                side_effect=[
                    enhance.requests.ReadTimeout("transient timeout"),
                    FakeGitHubResponse(),
                ],
            ) as request_get,
            patch.object(enhance.time, "sleep") as sleep,
        ):
            result = enhance.process_single_item(
                FakeChain(),
                FakeChain(),
                item,
                "Chinese",
                1,
            )

        self.assertEqual(result["code_stars"], 42)
        self.assertEqual(request_get.call_count, 2)
        self.assertEqual(request_get.call_args.kwargs["timeout"], (10, 30))
        self.assertEqual(
            request_get.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-token",
        )
        sleep.assert_called_once_with(1)

    def test_github_metadata_raises_after_timeout_retries_are_exhausted(self):
        with (
            patch.object(
                enhance.requests,
                "get",
                side_effect=enhance.requests.ReadTimeout("persistent timeout"),
            ) as request_get,
            patch.object(enhance.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed after 4 attempts"):
                enhance.fetch_github_repo_metadata("example", "robot-code", None)

        self.assertEqual(request_get.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4)])

    def test_github_metadata_retries_transient_http_status(self):
        with (
            patch.object(
                enhance.requests,
                "get",
                side_effect=[
                    FakeGitHubResponse(status_code=503),
                    FakeGitHubResponse(),
                ],
            ) as request_get,
            patch.object(enhance.time, "sleep") as sleep,
        ):
            metadata = enhance.fetch_github_repo_metadata(
                "example",
                "robot-code",
                None,
            )

        self.assertEqual(metadata["code_stars"], 42)
        self.assertEqual(request_get.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_github_metadata_does_not_retry_non_transient_http_status(self):
        with (
            patch.object(
                enhance.requests,
                "get",
                return_value=FakeGitHubResponse(status_code=403),
            ) as request_get,
            patch.object(enhance.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(enhance.requests.HTTPError, "HTTP 403"):
                enhance.fetch_github_repo_metadata(
                    "example",
                    "robot-code",
                    None,
                )

        request_get.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
