import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import MaterialInfo, VideoConcatMode
from app.services import material


PEXELS_SECRET = "secret-pexels-key-do-not-leak"
PIXABAY_SECRET = "secret-pixabay-key-do-not-leak"
TERM_SSD = "SSD storage aesthetic"
TERM_SSD_SIMPLE = "SSD storage"


def _clip(url, provider="pexels", duration=8):
    item = MaterialInfo()
    item.provider = provider
    item.url = url
    item.duration = duration
    return item


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _LogCapture:
    def __init__(self):
        self.lines = []
        self._handler_id = None

    def _sink(self, message):
        self.lines.append(message.record["message"])

    def __enter__(self):
        self._handler_id = logger.add(self._sink, format="{message}")
        return self

    def __exit__(self, exc_type, exc, tb):
        logger.remove(self._handler_id)

    def text(self):
        return "\n".join(self.lines)


def _fake_save(video_url, save_dir="", search_term="", thumbnail_url="", preview_images=None):
    return f"C:\\tmp\\cached\\{video_url.rsplit('/', 1)[-1]}"


class TestSimplifySearchTerm(unittest.TestCase):
    def test_ssd_storage_aesthetic_one_variant(self):
        self.assertEqual(material.simplify_search_term(TERM_SSD), TERM_SSD_SIMPLE)

    def test_single_word_has_no_variant(self):
        self.assertEqual(material.simplify_search_term("money"), "")


class TestDownloadVideosFallback(unittest.TestCase):
    def _run(
        self,
        search_terms,
        pexels,
        pixabay,
        pexels_key=True,
        pixabay_key=True,
        source="pexels",
    ):
        query_order = []

        def fake_pexels(search_term, minimum_duration, video_aspect=None):
            query_order.append(("pexels", search_term))
            result = pexels(search_term) if callable(pexels) else pexels.get(search_term, [])
            if isinstance(result, Exception):
                raise result
            return result

        def fake_pixabay(search_term, minimum_duration, video_aspect=None):
            query_order.append(("pixabay", search_term))
            result = pixabay(search_term) if callable(pixabay) else pixabay.get(search_term, [])
            if isinstance(result, Exception):
                raise result
            return result

        def fake_has_key(cfg_key):
            if cfg_key == "pixabay_api_keys":
                return pixabay_key
            if cfg_key == "pexels_api_keys":
                return pexels_key
            return True

        logs = _LogCapture()
        with logs, patch(
            "app.services.material.search_videos_pexels", side_effect=fake_pexels
        ), patch(
            "app.services.material.search_videos_pixabay", side_effect=fake_pixabay
        ), patch(
            "app.services.material.save_video", side_effect=_fake_save
        ), patch(
            "app.services.material.has_api_key", side_effect=fake_has_key
        ), patch(
            "app.services.pinterest.is_enabled", return_value=False
        ):
            paths = material.download_videos(
                task_id="t-fallback",
                search_terms=search_terms,
                source=source,
                audio_duration=10,
                max_clip_duration=5,
                video_contact_mode=VideoConcatMode.sequential,
            )
        return paths, query_order, logs.text()

    def test_primary_pexels_hit_skips_pixabay(self):
        paths, query_order, _ = self._run(
            search_terms=["money"],
            pexels={"money": [_clip("https://pexels.example/hit.mp4")]},
            pixabay={"money": [_clip("https://pixabay.example/unused.mp4", "pixabay")]},
        )
        self.assertEqual(query_order, [("pexels", "money")])
        self.assertEqual(paths, ["C:\\tmp\\cached\\hit.mp4"])

    def test_pexels_empty_returns_pixabay_hit(self):
        paths, query_order, _ = self._run(
            search_terms=["money"],
            pexels={"money": []},
            pixabay={"money": [_clip("https://pixabay.example/hit.mp4", "pixabay")]},
        )
        self.assertEqual(query_order, [("pexels", "money"), ("pixabay", "money")])
        self.assertEqual(paths, ["C:\\tmp\\cached\\hit.mp4"])

    def test_missing_pixabay_key_skips_fallback(self):
        paths, query_order, logs = self._run(
            search_terms=["money"],
            pexels={"money": []},
            pixabay={"money": [_clip("https://pixabay.example/hit.mp4", "pixabay")]},
            pixabay_key=False,
        )
        self.assertEqual(query_order, [("pexels", "money")])
        self.assertEqual(paths, [])
        self.assertIn("pixabay skipped (no key)", logs)
        self.assertNotIn(PIXABAY_SECRET, logs)
        self.assertNotIn(PEXELS_SECRET, logs)

    def test_missing_pexels_key_returns_pixabay_hit(self):
        paths, query_order, logs = self._run(
            search_terms=["money"],
            pexels={"money": [_clip("https://pexels.example/unused.mp4")]},
            pixabay={"money": [_clip("https://pixabay.example/hit.mp4", "pixabay")]},
            pexels_key=False,
            source="pexels",
        )
        self.assertEqual(query_order, [("pixabay", "money")])
        self.assertEqual(paths, ["C:\\tmp\\cached\\hit.mp4"])
        self.assertIn("pexels skipped (no key)", logs)
        self.assertNotIn(PIXABAY_SECRET, logs)
        self.assertNotIn(PEXELS_SECRET, logs)

    def test_missing_pixabay_key_returns_pexels_hit(self):
        paths, query_order, logs = self._run(
            search_terms=["money"],
            pexels={"money": [_clip("https://pexels.example/hit.mp4")]},
            pixabay={"money": [_clip("https://pixabay.example/unused.mp4", "pixabay")]},
            pixabay_key=False,
            source="pixabay",
        )
        self.assertEqual(query_order, [("pexels", "money")])
        self.assertEqual(paths, ["C:\\tmp\\cached\\hit.mp4"])
        self.assertIn("pixabay skipped (no key)", logs)
        self.assertNotIn(PIXABAY_SECRET, logs)
        self.assertNotIn(PEXELS_SECRET, logs)

    def test_ssd_storage_aesthetic_uses_simplified_variant(self):
        self.assertEqual(material.simplify_search_term(TERM_SSD), TERM_SSD_SIMPLE)
        paths, query_order, _ = self._run(
            search_terms=[TERM_SSD],
            pexels={
                TERM_SSD: [],
                TERM_SSD_SIMPLE: [_clip("https://pexels.example/simple.mp4")],
            },
            pixabay={TERM_SSD: [], TERM_SSD_SIMPLE: []},
        )
        self.assertEqual(
            query_order,
            [
                ("pexels", TERM_SSD),
                ("pixabay", TERM_SSD),
                ("pexels", TERM_SSD_SIMPLE),
            ],
        )
        self.assertEqual(paths, ["C:\\tmp\\cached\\simple.mp4"])

    def test_both_providers_empty_returns_empty(self):
        paths, query_order, _ = self._run(
            search_terms=[TERM_SSD],
            pexels={TERM_SSD: [], TERM_SSD_SIMPLE: []},
            pixabay={TERM_SSD: [], TERM_SSD_SIMPLE: []},
        )
        self.assertEqual(
            query_order,
            [
                ("pexels", TERM_SSD),
                ("pixabay", TERM_SSD),
                ("pexels", TERM_SSD_SIMPLE),
                ("pixabay", TERM_SSD_SIMPLE),
            ],
        )
        self.assertEqual(paths, [])

    def test_http_disable_codes_not_retried(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                def pexels(_term):
                    return material.StockProviderDisabled("pexels", status)

                def pixabay(term):
                    if term == "beta":
                        return [_clip("https://pixabay.example/beta.mp4", "pixabay")]
                    return []

                paths, query_order, logs = self._run(
                    search_terms=["alpha", "beta"],
                    pexels=pexels,
                    pixabay=pixabay,
                )
                self.assertEqual(
                    query_order,
                    [
                        ("pexels", "alpha"),
                        ("pixabay", "alpha"),
                        ("pixabay", "beta"),
                    ],
                    msg=f"HTTP {status} query order",
                )
                self.assertEqual(paths, ["C:\\tmp\\cached\\beta.mp4"])
                self.assertNotIn(PEXELS_SECRET, logs)
                self.assertNotIn(PIXABAY_SECRET, logs)


class TestSearchProviderHttpDisable(unittest.TestCase):
    def test_api_keys_not_in_logs_or_exceptions(self):
        captured = []
        for provider, fn, secret in (
            ("pexels", material.search_videos_pexels, PEXELS_SECRET),
            ("pixabay", material.search_videos_pixabay, PIXABAY_SECRET),
        ):
            for status in (401, 403, 429):
                logs = _LogCapture()
                with logs, patch(
                    "app.services.material.get_api_key", return_value=secret
                ), patch(
                    "app.services.material.requests.get",
                    return_value=_FakeResponse(status),
                ):
                    with self.assertRaises(material.StockProviderDisabled) as ctx:
                        fn(search_term="money", minimum_duration=5)
                blob = "\n".join(
                    [logs.text(), str(ctx.exception), repr(ctx.exception)]
                )
                captured.append((provider, status, blob))
                self.assertNotIn(secret, blob)
                self.assertNotIn(PEXELS_SECRET, blob)
                self.assertNotIn(PIXABAY_SECRET, blob)
                self.assertEqual(ctx.exception.provider, provider)
                self.assertEqual(ctx.exception.status_code, status)

        self.assertEqual(len(captured), 6)


if __name__ == "__main__":
    unittest.main()
