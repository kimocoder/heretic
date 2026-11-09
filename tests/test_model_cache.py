# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
import urllib.error
from http.client import HTTPMessage
from pathlib import Path
from unittest.mock import patch

from heretic.model import _MAX_HUB_PROBES, ensure_model_cached


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers


class _HubError(Exception):
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = _Response(status_code, headers or {})


class EnsureModelCachedTests(unittest.TestCase):
    def test_retries_rate_limit_before_modelscope(self) -> None:
        calls = 0

        def model_info(*_args: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _HubError(429, {"ratelimit": '"api";r=0;t=42'})
            return object()

        with (
            patch("heretic.model._find_complete_hf_snapshot", return_value=None),
            patch("heretic.model.huggingface_hub.model_info", side_effect=model_info),
            patch("heretic.model._download_from_modelscope") as mirror,
            patch("heretic.model.time.sleep") as sleep,
        ):
            resolved = ensure_model_cached("tiny-random/gpt-oss", "abc")

        self.assertEqual(resolved, "tiny-random/gpt-oss")
        self.assertEqual(calls, 2)
        mirror.assert_not_called()
        sleep.assert_called_once_with(43.0)

    def test_modelscope_404_reraises_hub_error(self) -> None:
        def model_info(*_args: object, **_kwargs: object) -> object:
            raise _HubError(429, {"Retry-After": "1"})

        mirror_error = urllib.error.HTTPError(
            "https://www.modelscope.cn/api/v1/models/tiny-random/gpt-oss/repo/files",
            404,
            "Not Found",
            HTTPMessage(),
            None,
        )

        with (
            patch("heretic.model._find_complete_hf_snapshot", return_value=None),
            patch(
                "heretic.model.huggingface_hub.model_info", side_effect=model_info
            ) as info,
            patch("heretic.model._download_from_modelscope", side_effect=mirror_error),
            patch("heretic.model.time.sleep") as sleep,
            self.assertRaises(_HubError) as raised,
        ):
            ensure_model_cached("tiny-random/gpt-oss")

        self.assertEqual(info.call_count, _MAX_HUB_PROBES)
        self.assertEqual(sleep.call_count, _MAX_HUB_PROBES - 1)
        cause = raised.exception.__cause__
        self.assertIsInstance(cause, urllib.error.HTTPError)
        assert isinstance(cause, urllib.error.HTTPError)
        cause.close()

    def test_missing_repo_uses_modelscope_without_retry(self) -> None:
        with (
            patch("heretic.model._find_complete_hf_snapshot", return_value=None),
            patch(
                "heretic.model.huggingface_hub.model_info",
                side_effect=_HubError(404),
            ) as info,
            patch(
                "heretic.model._download_from_modelscope",
                return_value=Path("/tmp/modelscope-snapshot"),
            ) as mirror,
            patch("heretic.model.time.sleep") as sleep,
        ):
            resolved = ensure_model_cached("org/missing-on-hub")

        self.assertEqual(resolved, "/tmp/modelscope-snapshot")
        self.assertEqual(info.call_count, 1)
        mirror.assert_called_once_with("org/missing-on-hub")
        sleep.assert_not_called()

    def test_complete_local_snapshot_skips_hub(self) -> None:
        with (
            patch(
                "heretic.model._find_complete_hf_snapshot",
                return_value=Path("/tmp/hf-snapshot"),
            ),
            patch("heretic.model.huggingface_hub.model_info") as info,
            patch("heretic.model._download_from_modelscope") as mirror,
        ):
            resolved = ensure_model_cached("org/model")

        self.assertEqual(resolved, "/tmp/hf-snapshot")
        info.assert_not_called()
        mirror.assert_not_called()
