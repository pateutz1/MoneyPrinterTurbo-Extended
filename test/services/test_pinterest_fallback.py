import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.services import material
from app.services import pinterest as pinterest_service


def _clip(url, provider="pexels", duration=8):
    item = MaterialInfo()
    item.provider = provider
    item.url = url
    item.duration = duration
    return item


def _pin(pin_id, duration=0):
    return _clip(f"https://www.pinterest.com/pin/{pin_id}/", "pinterest", duration)


def _slow_write_worker(target_path: str, delay: float, result_queue: mp.Queue) -> None:
    time.sleep(delay)
    Path(target_path).write_text("late-write", encoding="utf-8")
    result_queue.put("done")


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
    return f"C:\\tmp\\cached\\{video_url.rsplit('/', 1)[-1] or 'clip.mp4'}"


def _fake_pinterest_download(
    pin_url, save_dir="", video_aspect=None, minimum_duration=1
):
    return (
        pinterest_service.cache_video_path(pin_url, save_dir or r"C:\tmp\cached"),
        8.0,
    )


class TestPinUrlAllowlist(unittest.TestCase):
    def test_accepts_https_pin_urls(self):
        self.assertEqual(
            pinterest_service.normalize_pin_url(
                "https://www.pinterest.com/pin/123456789/"
            ),
            "https://www.pinterest.com/pin/123456789/",
        )
        self.assertTrue(
            pinterest_service.is_allowed_pin_url("https://pinterest.com/pin/99")
        )

    def test_rejects_non_pin_hosts_and_schemes(self):
        rejected = [
            "http://www.pinterest.com/pin/1/",
            "https://evil.example/pin/1/",
            "https://www.pinterest.com.evil.example/pin/1/",
            "https://www.pinterest.com/pin/abc/",
            "https://www.pinterest.com/search/videos/?q=ssd",
            "file:///C:/tmp/pin.mp4",
            "javascript:alert(1)",
            "",
        ]
        for url in rejected:
            self.assertFalse(pinterest_service.is_allowed_pin_url(url), url)


class TestProcessTimeout(unittest.TestCase):
    def test_timeout_terminates_worker_and_prevents_late_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = os.path.join(temp_dir, "late.txt")
            result, timed_out = pinterest_service.run_process_worker(
                _slow_write_worker,
                (target, 2.0),
                0.2,
                default=None,
                cleanup_dir=temp_dir,
            )
            time.sleep(0.3)
            self.assertTrue(timed_out)
            self.assertIsNone(result)
            self.assertFalse(os.path.isfile(target))

    def test_windows_timeout_uses_taskkill_with_shell_false(self):
        process = MagicMock()
        process.is_alive.side_effect = [True, False]
        process.pid = 4242
        with patch("app.services.pinterest.sys.platform", "win32"), patch(
            "app.services.pinterest.subprocess.run",
            return_value=MagicMock(returncode=0),
        ) as run_mock, patch(
            "app.services.pinterest._terminate_process"
        ) as fallback_mock:
            pinterest_service._terminate_process_tree(process)
        run_mock.assert_called_once_with(
            ["taskkill", "/PID", "4242", "/T", "/F"],
            shell=False,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        fallback_mock.assert_not_called()
        process.join.assert_called()


class TestProcessQueueHandling(unittest.TestCase):
    def test_run_process_worker_reads_result_without_empty(self):
        ctx = MagicMock()
        queue = MagicMock()
        queue.get.return_value = ["pin-a"]
        ctx.Queue.return_value = queue
        process = MagicMock()
        process.is_alive.return_value = False
        process.exitcode = 0
        ctx.Process.return_value = process

        with patch("app.services.pinterest.mp.get_context", return_value=ctx):
            result, timed_out = pinterest_service.run_process_worker(
                _slow_write_worker,
                ("unused", 0.01),
                1.0,
                default=[],
            )

        self.assertFalse(timed_out)
        self.assertEqual(result, ["pin-a"])
        queue.get.assert_called_once_with(timeout=pinterest_service._QUEUE_GET_TIMEOUT)
        queue.close.assert_called_once()
        queue.join_thread.assert_called_once()

    def test_run_process_worker_closes_queue_on_timeout(self):
        ctx = MagicMock()
        queue = MagicMock()
        ctx.Queue.return_value = queue
        process = MagicMock()
        process.is_alive.return_value = True
        process.pid = 999
        ctx.Process.return_value = process

        with patch("app.services.pinterest.mp.get_context", return_value=ctx), patch(
            "app.services.pinterest._terminate_process_tree"
        ):
            result, timed_out = pinterest_service.run_process_worker(
                _slow_write_worker,
                ("unused", 0.01),
                0.1,
                default=[],
            )

        self.assertTrue(timed_out)
        self.assertEqual(result, [])
        queue.close.assert_called_once()
        queue.join_thread.assert_called_once()


class TestDownloadLimits(unittest.TestCase):
    def test_progress_hook_rejects_oversized_transfer(self):
        hook = pinterest_service.make_download_progress_hook(1024)
        with self.assertRaises(pinterest_service.PinterestDownloadSizeExceeded):
            hook({"status": "downloading", "downloaded_bytes": 2048})

    def test_build_ytdlp_opts_includes_size_limits(self):
        opts = pinterest_service.build_ytdlp_opts("out.%(ext)s", 15, 4096)
        self.assertEqual(opts["max_filesize"], 4096)
        self.assertEqual(len(opts["progress_hooks"]), 1)


class TestValidateDownloadedClip(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="pinterest-validate-")

    def tearDown(self):
        pinterest_service._safe_remove(self.temp_dir)

    def _write_file(self, name: str, size: int) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "wb") as handle:
            handle.write(b"x" * size)
        return path

    def _mock_clip(self, duration=8.0, fps=30.0, width=1080, height=1920):
        clip = MagicMock()
        clip.duration = duration
        clip.fps = fps
        clip.w = width
        clip.h = height
        return clip

    def test_rejects_oversized_file(self):
        path = self._write_file("big.mp4", 2048)
        with patch(
            "app.services.pinterest.max_file_bytes", return_value=1024
        ), patch("app.services.pinterest.VideoFileClip") as clip_mock:
            ok, duration, reason = pinterest_service.validate_downloaded_clip(
                path, VideoAspect.portrait, minimum_duration=5
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "oversized file")
        self.assertEqual(duration, 0.0)
        clip_mock.assert_not_called()

    def test_rejects_short_clip(self):
        path = self._write_file("short.mp4", 128)
        with patch(
            "app.services.pinterest.VideoFileClip",
            return_value=self._mock_clip(duration=2.0),
        ):
            ok, duration, reason = pinterest_service.validate_downloaded_clip(
                path, VideoAspect.portrait, minimum_duration=5
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "short clip")
        self.assertEqual(duration, 2.0)

    def test_rejects_wrong_aspect(self):
        path = self._write_file("landscape.mp4", 128)
        with patch(
            "app.services.pinterest.VideoFileClip",
            return_value=self._mock_clip(width=1920, height=1080),
        ):
            ok, duration, reason = pinterest_service.validate_downloaded_clip(
                path, VideoAspect.portrait, minimum_duration=5
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "aspect mismatch")

    def test_accepts_proportional_portrait_resolution(self):
        path = self._write_file("hd-portrait.mp4", 128)
        clip = self._mock_clip(width=720, height=1280)
        with patch("app.services.pinterest.VideoFileClip", return_value=clip):
            ok, duration, reason = pinterest_service.validate_downloaded_clip(
                path, VideoAspect.portrait, minimum_duration=5
            )
        self.assertTrue(ok)
        self.assertEqual(duration, 8.0)
        self.assertEqual(reason, "")

    def test_accepts_valid_portrait_clip(self):
        path = self._write_file("good.mp4", 128)
        clip = self._mock_clip()
        with patch("app.services.pinterest.VideoFileClip", return_value=clip):
            ok, duration, reason = pinterest_service.validate_downloaded_clip(
                path, VideoAspect.portrait, minimum_duration=5
            )
        self.assertTrue(ok)
        self.assertEqual(duration, 8.0)
        self.assertEqual(reason, "")
        clip.close.assert_called_once()


class TestDownloadPinVideo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="pinterest-dl-test-")

    def tearDown(self):
        pinterest_service._safe_remove(self.temp_dir)

    def test_atomic_publish_replaces_existing_destination(self):
        final_path = os.path.join(self.temp_dir, "vid-test.mp4")
        source_path = os.path.join(self.temp_dir, "source.mp4")
        Path(final_path).write_bytes(b"old")
        Path(source_path).write_bytes(b"new-content")
        pinterest_service._atomic_publish(source_path, final_path)
        self.assertFalse(os.path.exists(source_path))
        self.assertEqual(Path(final_path).read_bytes(), b"new-content")

    def test_publishes_expected_cache_name(self):
        pin_url = "https://www.pinterest.com/pin/123456789/"
        expected = pinterest_service.cache_video_path(pin_url, self.temp_dir)
        staging = os.path.join(self.temp_dir, "worker", "validated.mp4")
        os.makedirs(os.path.dirname(staging), exist_ok=True)
        with open(staging, "wb") as handle:
            handle.write(b"x" * 128)

        def fake_worker(*args, **kwargs):
            return {"staging_path": staging}, False

        with patch(
            "app.services.pinterest.optional_deps_available", return_value=True
        ), patch(
            "app.services.pinterest.run_process_worker", side_effect=fake_worker
        ), patch(
            "app.services.pinterest.validate_downloaded_clip",
            return_value=(True, 7.5, ""),
        ), patch(
            "app.services.pinterest.tempfile.mkdtemp",
            return_value=os.path.join(self.temp_dir, "worker"),
        ):
            path, duration = pinterest_service.download_pin_video(
                pin_url,
                save_dir=self.temp_dir,
                video_aspect=VideoAspect.portrait,
                minimum_duration=5,
            )

        self.assertEqual(path, expected)
        self.assertEqual(duration, 7.5)
        self.assertTrue(os.path.isfile(expected))
        self.assertFalse(os.path.isdir(os.path.join(self.temp_dir, "worker")))

    def test_cleans_partial_files_on_validation_failure(self):
        pin_url = "https://www.pinterest.com/pin/222/"
        worker_dir = os.path.join(self.temp_dir, "worker")
        staging = os.path.join(worker_dir, "validated.mp4")
        os.makedirs(worker_dir, exist_ok=True)
        with open(staging, "wb") as handle:
            handle.write(b"x" * 64)
        partial = os.path.join(worker_dir, "download.part")
        with open(partial, "wb") as handle:
            handle.write(b"partial")

        def fake_worker(*args, **kwargs):
            return {"staging_path": staging}, False

        with patch(
            "app.services.pinterest.optional_deps_available", return_value=True
        ), patch(
            "app.services.pinterest.run_process_worker", side_effect=fake_worker
        ), patch(
            "app.services.pinterest.validate_downloaded_clip",
            return_value=(False, 0.0, "short clip"),
        ), patch(
            "app.services.pinterest.tempfile.mkdtemp", return_value=worker_dir
        ):
            path, duration = pinterest_service.download_pin_video(
                pin_url,
                save_dir=self.temp_dir,
                video_aspect=VideoAspect.portrait,
                minimum_duration=5,
            )

        self.assertEqual(path, "")
        self.assertEqual(duration, 0.0)
        self.assertFalse(os.path.exists(staging))
        self.assertFalse(os.path.exists(partial))
        self.assertFalse(os.path.isdir(worker_dir))


class TestPinterestFallback(unittest.TestCase):
    def _run(
        self,
        search_terms,
        pexels,
        pixabay,
        enabled=True,
        deps=True,
        pinterest_items=None,
        pinterest_error=None,
        max_dl=2,
        max_attempts=10,
        fake_download=None,
    ):
        query_order = []
        pin_calls = []
        download_calls = []

        def fake_pexels(search_term, minimum_duration, video_aspect=None):
            query_order.append(("pexels", search_term))
            return pexels.get(search_term, [])

        def fake_pixabay(search_term, minimum_duration, video_aspect=None):
            query_order.append(("pixabay", search_term))
            return pixabay.get(search_term, [])

        def fake_search(search_term, minimum_duration, video_aspect=None):
            pin_calls.append(search_term)
            if pinterest_error:
                raise pinterest_error
            items = pinterest_items
            if callable(pinterest_items):
                items = pinterest_items(search_term)
            return list(items or [])

        def fake_download_impl(pin_url, save_dir="", video_aspect=None, minimum_duration=1):
            download_calls.append(pin_url)
            if fake_download:
                return fake_download(pin_url, save_dir, video_aspect, minimum_duration)
            return _fake_pinterest_download(
                pin_url, save_dir, video_aspect, minimum_duration
            )

        logs = _LogCapture()
        with logs, patch(
            "app.services.material.search_videos_pexels", side_effect=fake_pexels
        ), patch(
            "app.services.material.search_videos_pixabay", side_effect=fake_pixabay
        ), patch(
            "app.services.material.save_video", side_effect=_fake_save
        ), patch(
            "app.services.material.has_api_key", return_value=True
        ), patch(
            "app.services.pinterest.is_enabled", return_value=enabled
        ), patch(
            "app.services.pinterest.optional_deps_available", return_value=deps
        ), patch(
            "app.services.pinterest.max_downloads", return_value=max_dl
        ), patch(
            "app.services.pinterest.max_attempts", return_value=max_attempts
        ), patch(
            "app.services.pinterest.search_videos", side_effect=fake_search
        ), patch(
            "app.services.pinterest.download_pin_video",
            side_effect=fake_download_impl,
        ):
            paths = material.download_videos(
                task_id="t-pinterest",
                search_terms=search_terms,
                source="pexels",
                audio_duration=30,
                max_clip_duration=5,
                video_contact_mode=VideoConcatMode.sequential,
            )
        return paths, query_order, pin_calls, download_calls, logs.text()

    def test_disabled_flag_skips_pinterest(self):
        paths, _, pin_calls, download_calls, _ = self._run(
            search_terms=["money"],
            pexels={"money": []},
            pixabay={"money": []},
            enabled=False,
            pinterest_items=[_pin("111")],
        )
        self.assertEqual(paths, [])
        self.assertEqual(pin_calls, [])
        self.assertEqual(download_calls, [])

    def test_enabled_after_stock_empty_calls_once_per_term(self):
        paths, _, pin_calls, download_calls, _ = self._run(
            search_terms=["alpha", "beta"],
            pexels={"alpha": [], "beta": []},
            pixabay={"alpha": [], "beta": []},
            pinterest_items=lambda term: [_pin("111" if term == "alpha" else "222")],
            max_dl=2,
        )
        self.assertEqual(pin_calls, ["alpha", "beta"])
        self.assertEqual(len(download_calls), 2)
        self.assertEqual(len(paths), 2)

    def test_stock_hit_skips_pinterest(self):
        paths, _, pin_calls, download_calls, _ = self._run(
            search_terms=["money"],
            pexels={"money": [_clip("https://pexels.example/hit.mp4")]},
            pixabay={"money": []},
            pinterest_items=[_pin("111")],
        )
        self.assertEqual(pin_calls, [])
        self.assertEqual(download_calls, [])
        self.assertEqual(paths, ["C:\\tmp\\cached\\hit.mp4"])

    def test_search_returns_unknown_duration_metadata(self):
        with patch(
            "app.services.pinterest.is_enabled", return_value=True
        ), patch(
            "app.services.pinterest.optional_deps_available", return_value=True
        ), patch(
            "app.services.pinterest.search_pin_urls",
            return_value=["https://www.pinterest.com/pin/111/"],
        ):
            items = pinterest_service.search_videos("money", minimum_duration=5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].duration, 0)

    def test_global_download_cap_across_terms(self):
        pin_ids = {"alpha": "101", "beta": "202", "gamma": "303"}
        paths, _, pin_calls, download_calls, logs = self._run(
            search_terms=["alpha", "beta", "gamma"],
            pexels={"alpha": [], "beta": [], "gamma": []},
            pixabay={"alpha": [], "beta": [], "gamma": []},
            pinterest_items=lambda term: [
                _pin(pin_ids[term]),
                _pin(str(int(pin_ids[term]) + 1)),
            ],
            max_dl=1,
            max_attempts=10,
        )
        self.assertEqual(pin_calls, ["alpha", "beta", "gamma"])
        self.assertEqual(len(download_calls), 1)
        self.assertEqual(len(paths), 1)
        self.assertIn("pinterest download cap reached for this operation", logs)

    def test_global_attempt_cap_stops_after_failed_downloads(self):
        paths, _, _, download_calls, logs = self._run(
            search_terms=["alpha"],
            pexels={"alpha": []},
            pixabay={"alpha": []},
            pinterest_items=[
                _pin("101"),
                _pin("102"),
                _pin("103"),
                _pin("104"),
            ],
            max_dl=5,
            max_attempts=2,
            fake_download=lambda *args, **kwargs: ("", 0.0),
        )
        self.assertEqual(len(download_calls), 2)
        self.assertEqual(paths, [])
        self.assertIn("pinterest attempt cap reached for this operation", logs)

    def test_actual_duration_used_for_accounting(self):
        def fake_download(pin_url, save_dir="", video_aspect=None, minimum_duration=1):
            return (
                pinterest_service.cache_video_path(pin_url, save_dir or r"C:\tmp\cached"),
                12.0,
            )

        paths, _, _, download_calls, _ = self._run(
            search_terms=["money"],
            pexels={"money": []},
            pixabay={"money": []},
            pinterest_items=[_pin("111"), _pin("222")],
            max_dl=1,
            fake_download=fake_download,
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(download_calls), 1)

    def test_dedup_across_terms(self):
        shared = _pin("111")
        paths, _, pin_calls, download_calls, _ = self._run(
            search_terms=["alpha", "beta"],
            pexels={"alpha": [], "beta": []},
            pixabay={"alpha": [], "beta": []},
            pinterest_items=[shared],
            max_dl=2,
        )
        self.assertEqual(pin_calls, ["alpha", "beta"])
        self.assertEqual(len(download_calls), 1)
        self.assertEqual(len(paths), 1)

    def test_missing_deps_are_non_fatal(self):
        logs = _LogCapture()
        with logs, patch(
            "app.services.pinterest.is_enabled", return_value=True
        ), patch(
            "app.services.pinterest.optional_deps_available", return_value=False
        ), patch(
            "app.services.pinterest.search_pin_urls"
        ) as search_mock:
            items = pinterest_service.search_videos("money", minimum_duration=5)
        self.assertEqual(items, [])
        search_mock.assert_not_called()
        self.assertIn("pinterest skipped (missing deps)", logs.text())

    def test_pinterest_error_is_non_fatal(self):
        paths, _, pin_calls, download_calls, logs = self._run(
            search_terms=["money"],
            pexels={"money": []},
            pixabay={"money": []},
            pinterest_error=RuntimeError("extractor exploded with secret-token"),
        )
        self.assertEqual(paths, [])
        self.assertEqual(pin_calls, ["money"])
        self.assertEqual(download_calls, [])
        self.assertIn("pinterest fallback failed: unexpected error", logs)
        self.assertNotIn("secret-token", logs)
        self.assertNotIn("extractor exploded", logs)

    def test_cache_path_uses_md5_layout(self):
        pin_url = "https://www.pinterest.com/pin/123456789/"
        path = pinterest_service.cache_video_path(pin_url, r"C:\tmp\cached")
        expected_hash = __import__("app.utils.utils", fromlist=["md5"]).md5(
            "https://www.pinterest.com/pin/123456789/"
        )
        self.assertEqual(path, rf"C:\tmp\cached\vid-{expected_hash}.mp4")


if __name__ == "__main__":
    unittest.main()
