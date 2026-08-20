import asyncio
import glob
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, List, Optional, Tuple
from urllib.parse import quote, urlparse

from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils

_PIN_PATH_RE = re.compile(r"^/pin/(\d+)/?$")
_PIN_HOSTS = {"www.pinterest.com", "pinterest.com"}
_MISSING_DEPS_LOGGED = False
_ARTIFACT_SUFFIXES = (".part", ".temp", ".tmp", ".ytdl")
_MIN_RESOLUTION = {
    VideoAspect.portrait.value: (360, 640),
    VideoAspect.landscape.value: (640, 360),
    VideoAspect.square.value: (360, 360),
}
_QUEUE_GET_TIMEOUT = 2.0


class PinterestDownloadSizeExceeded(Exception):
    pass


def is_enabled() -> bool:
    return bool(config.app.get("enable_pinterest", False))


def _int_setting(key: str, default: int, minimum: int, maximum: int) -> int:
    raw = config.app.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def max_pins() -> int:
    return _int_setting("pinterest_max_pins", 5, 1, 10)


def max_downloads() -> int:
    return _int_setting("pinterest_max_downloads", 2, 1, 5)


def max_attempts() -> int:
    return _int_setting("pinterest_max_attempts", 3, 1, 10)


def timeout_seconds() -> int:
    return _int_setting("pinterest_timeout_seconds", 45, 10, 90)


def scroll_count() -> int:
    return _int_setting("pinterest_scroll_count", 2, 1, 5)


def max_file_bytes() -> int:
    return _int_setting("pinterest_max_file_bytes", 52_428_800, 1_048_576, 104_857_600)


def optional_deps_available() -> bool:
    try:
        import playwright  # noqa: F401
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return True


def normalize_pin_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https":
        return ""
    host = (parsed.netloc or "").split("@")[-1].split(":")[0].lower()
    if host not in _PIN_HOSTS:
        return ""
    match = _PIN_PATH_RE.match(parsed.path or "")
    if not match:
        return ""
    return f"https://www.pinterest.com/pin/{match.group(1)}/"


def is_allowed_pin_url(url: str) -> bool:
    return bool(normalize_pin_url(url))


def cache_video_path(pin_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")
    normalized = normalize_pin_url(pin_url) or pin_url.split("?")[0]
    video_id = f"vid-{utils.md5(normalized)}"
    return os.path.join(save_dir, f"{video_id}.mp4")


def _aspect_ratio_key(width: int, height: int) -> Tuple[int, int]:
    width = abs(int(width))
    height = abs(int(height))
    if width <= 0 or height <= 0:
        return 0, 0

    def _gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    divisor = _gcd(width, height)
    return width // divisor, height // divisor


def _aspect_matches(width: int, height: int, video_aspect: VideoAspect) -> bool:
    aspect = VideoAspect(video_aspect)
    expected_w, expected_h = aspect.to_resolution()
    min_w, min_h = _MIN_RESOLUTION.get(aspect.value, (360, 640))
    width = int(width)
    height = int(height)

    if width < min_w or height < min_h:
        return False

    actual = _aspect_ratio_key(width, height)
    expected = _aspect_ratio_key(expected_w, expected_h)
    if actual != expected:
        return False

    if aspect == VideoAspect.portrait and width >= height:
        return False
    if aspect == VideoAspect.landscape and height >= width:
        return False
    if aspect == VideoAspect.square and width != height:
        return False
    return True


def _cleanup_download_artifacts(directory: str, keep_path: str = "") -> None:
    if not directory or not os.path.isdir(directory):
        return
    keep = os.path.abspath(keep_path) if keep_path else ""
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if keep and os.path.abspath(path) == keep:
            continue
        lowered = entry.lower()
        if lowered.endswith(".mp4") and keep:
            continue
        if (
            lowered.endswith(_ARTIFACT_SUFFIXES)
            or "." not in entry
            or lowered.endswith(".part")
        ):
            _safe_remove(path)
            continue
        if lowered.endswith(".mp4") and not keep:
            _safe_remove(path)


def _safe_remove(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
    except OSError:
        pass


def _atomic_publish(source_path: str, final_path: str) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    os.replace(source_path, final_path)


def _find_ytdlp_output(temp_dir: str) -> str:
    candidates = []
    for pattern in ("*.mp4", "*.*"):
        candidates.extend(glob.glob(os.path.join(temp_dir, pattern)))
    mp4_files = [
        path
        for path in candidates
        if path.lower().endswith(".mp4")
        and not path.lower().endswith(".part")
        and os.path.isfile(path)
    ]
    if not mp4_files:
        return ""
    mp4_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return mp4_files[0]


def validate_downloaded_clip(
    path: str,
    video_aspect: VideoAspect,
    minimum_duration: int,
    max_bytes: Optional[int] = None,
) -> Tuple[bool, float, str]:
    if not path or not os.path.isfile(path):
        return False, 0.0, "missing file"

    size = os.path.getsize(path)
    limit = max_bytes if max_bytes is not None else max_file_bytes()
    if size <= 0:
        return False, 0.0, "empty file"
    if size > limit:
        return False, 0.0, "oversized file"

    clip = None
    try:
        clip = VideoFileClip(path)
        duration = float(clip.duration or 0)
        fps = float(clip.fps or 0)
        width = int(getattr(clip, "w", 0) or 0)
        height = int(getattr(clip, "h", 0) or 0)
        if duration <= 0 or fps <= 0:
            return False, 0.0, "invalid timing"
        if duration < float(minimum_duration):
            return False, duration, "short clip"
        if width <= 0 or height <= 0:
            return False, duration, "invalid dimensions"
        if not _aspect_matches(width, height, video_aspect):
            return False, duration, "aspect mismatch"
        return True, duration, ""
    except Exception:
        return False, 0.0, "invalid video"
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def make_download_progress_hook(max_bytes: int) -> Callable[[dict], None]:
    def _hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        downloaded = int(status.get("downloaded_bytes") or 0)
        if downloaded > max_bytes:
            raise PinterestDownloadSizeExceeded("size limit exceeded")

    return _hook


def build_ytdlp_opts(
    outtmpl: str,
    socket_timeout: int,
    max_bytes: int,
) -> dict:
    return {
        "format": "best[ext=mp4]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
        "noplaylist": True,
        "socket_timeout": socket_timeout,
        "max_filesize": max_bytes,
        "progress_hooks": [make_download_progress_hook(max_bytes)],
    }


def _terminate_process(process: mp.Process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _terminate_process_tree(process: mp.Process) -> None:
    if not process.is_alive():
        return
    pid = process.pid
    if sys.platform == "win32" and pid:
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                shell=False,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            process.join(timeout=5)
            if process.is_alive() or completed.returncode not in (0, 128):
                _terminate_process(process)
            return
        except Exception:
            _terminate_process(process)
            return
    _terminate_process(process)


def _close_result_queue(result_queue: mp.Queue) -> None:
    try:
        result_queue.close()
    except Exception:
        pass
    try:
        result_queue.join_thread()
    except Exception:
        pass


def _read_result_queue(result_queue: mp.Queue, default=None):
    try:
        return result_queue.get(timeout=_QUEUE_GET_TIMEOUT)
    except Exception:
        return default


def run_process_worker(
    worker_fn,
    worker_args: tuple,
    timeout_seconds: float,
    default=None,
    cleanup_dir: str = "",
) -> Tuple[object, bool]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=worker_fn, args=(*worker_args, result_queue))
    timed_out = False
    try:
        process.start()
        process.join(timeout=timeout_seconds)
        timed_out = process.is_alive()
        if timed_out:
            _terminate_process_tree(process)
            if cleanup_dir:
                _cleanup_download_artifacts(cleanup_dir)
                _safe_remove(cleanup_dir)
            logger.warning("pinterest skipped: timeout")
            return default, True
        if process.exitcode not in (0, None) and cleanup_dir:
            _cleanup_download_artifacts(cleanup_dir)
            _safe_remove(cleanup_dir)
        return _read_result_queue(result_queue, default=default), False
    finally:
        _close_result_queue(result_queue)


def _log_missing_deps() -> None:
    global _MISSING_DEPS_LOGGED
    if not _MISSING_DEPS_LOGGED:
        logger.warning("pinterest skipped (missing deps)")
        _MISSING_DEPS_LOGGED = True


def _search_pins_worker(
    query: str,
    pin_limit: int,
    scrolls: int,
    page_timeout_ms: int,
    result_queue: mp.Queue,
) -> None:
    try:
        result_queue.put(
            asyncio.run(
                _search_pins_async(query, pin_limit, scrolls, page_timeout_ms)
            )
        )
    except Exception as exc:
        result_queue.put({"error": type(exc).__name__})


async def _search_pins_async(
    query: str, pin_limit: int, scrolls: int, page_timeout_ms: int
) -> List[str]:
    from playwright.async_api import async_playwright

    search_url = f"https://www.pinterest.com/search/videos/?q={quote(query)}"
    found: List[str] = []
    seen = set()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(search_url, wait_until="domcontentloaded", timeout=page_timeout_ms)
            for _ in range(scrolls):
                await page.evaluate("window.scrollBy(0, 1500)")
                await page.wait_for_timeout(400)
            hrefs = await page.eval_on_selector_all(
                'a[href*="/pin/"]',
                "els => els.map(el => el.href)",
            )
            for href in hrefs or []:
                pin_url = normalize_pin_url(href)
                if not pin_url or pin_url in seen:
                    continue
                seen.add(pin_url)
                found.append(pin_url)
                if len(found) >= pin_limit:
                    break
        finally:
            await browser.close()
    return found


def search_pin_urls(
    query: str, pin_limit: Optional[int] = None, seconds: Optional[int] = None
) -> List[str]:
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return []
    limit = pin_limit if pin_limit is not None else max_pins()
    budget = seconds if seconds is not None else timeout_seconds()
    page_timeout_ms = min(budget, 30) * 1000
    scrolls = scroll_count()

    result, timed_out = run_process_worker(
        _search_pins_worker,
        (cleaned, limit, scrolls, page_timeout_ms),
        budget,
        default=[],
    )
    if timed_out:
        return []
    if isinstance(result, dict) and result.get("error"):
        logger.warning(f"pinterest search failed: {result['error']}")
        return []
    return list(result or [])


def search_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    del minimum_duration, video_aspect
    if not is_enabled():
        return []
    if not optional_deps_available():
        _log_missing_deps()
        return []

    pins = search_pin_urls(search_term)
    items: List[MaterialInfo] = []
    for pin_url in pins[: max_pins()]:
        item = MaterialInfo()
        item.provider = "pinterest"
        item.url = pin_url
        item.duration = 0
        items.append(item)
    return items


def _download_pin_worker(
    normalized_pin_url: str,
    temp_dir: str,
    socket_timeout: int,
    max_bytes: int,
    result_queue: mp.Queue,
) -> None:
    temp_dir = os.path.abspath(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    sent = False
    try:
        import yt_dlp

        outtmpl = os.path.join(temp_dir, "download.%(ext)s")
        opts = build_ytdlp_opts(outtmpl, socket_timeout, max_bytes)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(normalized_pin_url, download=True)
        output_path = _find_ytdlp_output(temp_dir)
        if not output_path:
            result_queue.put({"error": "missing output"})
            sent = True
            return
        staging_path = os.path.join(temp_dir, "validated.mp4")
        os.replace(output_path, staging_path)
        _cleanup_download_artifacts(temp_dir, keep_path=staging_path)
        result_queue.put({"staging_path": staging_path})
        sent = True
    except Exception as exc:
        _cleanup_download_artifacts(temp_dir)
        result_queue.put({"error": type(exc).__name__})
        sent = True
    finally:
        if not sent:
            _cleanup_download_artifacts(temp_dir)


def download_pin_video(
    pin_url: str,
    save_dir: str = "",
    video_aspect: VideoAspect = VideoAspect.portrait,
    minimum_duration: int = 1,
) -> Tuple[str, float]:
    normalized = normalize_pin_url(pin_url)
    if not normalized:
        logger.warning("pinterest download failed: invalid url")
        return "", 0.0
    if not optional_deps_available():
        _log_missing_deps()
        return "", 0.0

    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")
    os.makedirs(save_dir, exist_ok=True)
    final_path = cache_video_path(normalized, save_dir)
    byte_limit = max_file_bytes()

    if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
        ok, duration, reason = validate_downloaded_clip(
            final_path, video_aspect, minimum_duration, max_bytes=byte_limit
        )
        if ok:
            logger.info(f"video already exists: {final_path}")
            return final_path, duration
        logger.warning(f"pinterest download failed: {reason}")
        _safe_remove(final_path)

    temp_dir = tempfile.mkdtemp(prefix="pinterest-dl-", dir=save_dir)
    try:
        result, timed_out = run_process_worker(
            _download_pin_worker,
            (normalized, temp_dir, min(30, timeout_seconds()), byte_limit),
            timeout_seconds(),
            default={"error": "timeout"},
            cleanup_dir=temp_dir,
        )
        if timed_out:
            _cleanup_download_artifacts(temp_dir)
            _safe_remove(temp_dir)
            return "", 0.0

        if not isinstance(result, dict):
            logger.warning("pinterest download failed: invalid worker result")
            return "", 0.0
        if result.get("error"):
            logger.warning(f"pinterest download failed: {result['error']}")
            return "", 0.0

        staging_path = result.get("staging_path", "")
        if not staging_path or not os.path.isfile(staging_path):
            logger.warning("pinterest download failed: missing output")
            return "", 0.0

        ok, duration, reason = validate_downloaded_clip(
            staging_path, video_aspect, minimum_duration, max_bytes=byte_limit
        )
        if not ok:
            logger.warning(f"pinterest download failed: {reason}")
            _safe_remove(staging_path)
            return "", 0.0

        publish_staging = os.path.join(temp_dir, "publish.mp4")
        os.replace(staging_path, publish_staging)
        _atomic_publish(publish_staging, final_path)
        return final_path, duration
    finally:
        _cleanup_download_artifacts(temp_dir)
        _safe_remove(temp_dir)


def probe_cached_duration(
    pin_url: str, save_dir: str = "", video_aspect: VideoAspect = VideoAspect.portrait
) -> float:
    path = cache_video_path(pin_url, save_dir)
    if not os.path.isfile(path):
        return 0.0
    ok, duration, _ = validate_downloaded_clip(path, video_aspect, minimum_duration=1)
    return duration if ok else 0.0
