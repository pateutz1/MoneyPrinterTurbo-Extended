import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from loguru import logger
from PIL import Image, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoConcatMode, VideoParams
from app.services import task as task_service
from app.services import thumbnail as thumbnail_service


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


def _params(**overrides):
    base = dict(
        video_subject="SSD Upgrade Guide",
        video_script="",
        video_terms=None,
        video_aspect="9:16",
        video_concat_mode="random",
        video_transition_mode="None",
        video_clip_duration=5,
        video_count=1,
        video_source="local",
        video_language="",
        voice_name="en-US-AriaNeural-Female",
        voice_volume=1.0,
        voice_rate=1.0,
        bgm_type="random",
        bgm_file="",
        bgm_volume=0.2,
        subtitle_enabled=True,
        subtitle_position="bottom",
        custom_position=70.0,
        font_name="MissingFont.ttc",
        text_fore_color="#FFFFFF",
        text_background_color=True,
        font_size=60,
        stroke_color="#000000",
        stroke_width=1.5,
        n_threads=2,
        paragraph_number=1,
    )
    base.update(overrides)
    return VideoParams(**base)


class TestThumbnailHelpers(unittest.TestCase):
    def test_output_naming(self):
        self.assertEqual(
            thumbnail_service.thumbnail_output_path(
                r"C:\tasks\demo\final-1.mp4"
            ),
            r"C:\tasks\demo\final-1-thumbnail.jpg",
        )
        self.assertEqual(
            thumbnail_service.thumbnail_output_path(
                r"C:\tasks\demo\final-2.mp4"
            ),
            r"C:\tasks\demo\final-2-thumbnail.jpg",
        )

    def test_frame_timestamp_is_one_third(self):
        self.assertEqual(thumbnail_service.frame_timestamp(30.0), 10.0)


class TestGenerateThumbnail(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(__file__).parent / "_thumb_tmp"
        self.temp_dir.mkdir(exist_ok=True)
        self.video_path = self.temp_dir / "final-1.mp4"
        self.video_path.write_bytes(b"fake-video")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for pattern in ("*.mp4", "*.jpg"):
            for path in self.temp_dir.glob(pattern):
                path.unlink(missing_ok=True)
        if self.temp_dir.exists():
            self.temp_dir.rmdir()

    def _mock_clip(self, duration=30.0, width=1080, height=1920):
        mock_clip = MagicMock()
        mock_clip.duration = duration
        mock_clip.get_frame.return_value = np.zeros(
            (height, width, 3), dtype=np.uint8
        )
        return mock_clip

    def test_enabled_generates_one_thumbnail(self):
        mock_clip = self._mock_clip()
        with patch(
            "app.services.thumbnail.VideoFileClip", return_value=mock_clip
        ), patch(
            "app.services.thumbnail._resolve_font",
            return_value=ImageFont.load_default(),
        ):
            result = thumbnail_service.generate_thumbnail(
                str(self.video_path),
                title="SSD Upgrade Guide",
                font_name="MissingFont.ttc",
            )

        output_path = thumbnail_service.thumbnail_output_path(str(self.video_path))
        self.assertEqual(result, output_path)
        self.assertTrue(os.path.isfile(output_path))
        with Image.open(output_path) as image:
            self.assertEqual(image.size, (1080, 1920))
        mock_clip.get_frame.assert_called_once_with(10.0)
        mock_clip.close.assert_called_once()

    def test_missing_font_uses_safe_fallback(self):
        mock_clip = self._mock_clip()
        with patch(
            "app.services.thumbnail.VideoFileClip", return_value=mock_clip
        ), patch(
            "app.services.thumbnail._resolve_font",
            return_value=ImageFont.load_default(),
        ) as resolve_mock:
            thumbnail_service.generate_thumbnail(
                str(self.video_path),
                title="SSD Upgrade Guide",
                font_name="MissingFont.ttc",
            )
        resolve_mock.assert_called_once_with("MissingFont.ttc")

    def test_thumbnail_error_is_non_fatal(self):
        mock_clip = self._mock_clip()
        mock_clip.get_frame.side_effect = RuntimeError("frame failure")
        logs = _LogCapture()
        with logs, patch(
            "app.services.thumbnail.VideoFileClip", return_value=mock_clip
        ):
            result = thumbnail_service.generate_thumbnail(str(self.video_path))

        self.assertIsNone(result)
        self.assertIn("thumbnail generation failed: RuntimeError", logs.text())
        self.assertNotIn("frame failure", logs.text())
        mock_clip.close.assert_called_once()

    def test_video_resources_are_closed_on_success(self):
        mock_clip = self._mock_clip()
        with patch(
            "app.services.thumbnail.VideoFileClip", return_value=mock_clip
        ), patch(
            "app.services.thumbnail._resolve_font",
            return_value=ImageFont.load_default(),
        ):
            thumbnail_service.generate_thumbnail(str(self.video_path))
        mock_clip.close.assert_called_once()


def _thumbnail_config_get(enabled):
    mock_config = MagicMock()

    def _get(key, default=None):
        if key == "enable_thumbnail":
            return enabled
        return default

    mock_config.app.get = _get
    return mock_config


class TestTaskThumbnailIntegration(unittest.TestCase):
    def test_disabled_flag_skips_thumbnail(self):
        params = _params(video_count=2)
        with patch(
            "app.services.task.config",
            _thumbnail_config_get(False),
        ), patch.object(
            task_service.sm.state, "update_task"
        ), patch.object(
            task_service.video, "combine_videos"
        ), patch.object(
            task_service.video, "generate_video"
        ), patch.object(
            task_service.thumbnail_service, "generate_thumbnail"
        ) as thumb_mock, patch(
            "app.services.task.utils.task_dir",
            side_effect=lambda task_id: rf"C:\tasks\{task_id}",
        ), patch(
            "app.services.task.path.join",
            side_effect=os.path.join,
        ), patch(
            "app.services.task.path.isfile", return_value=True
        ), patch(
            "app.services.task.path.getsize", return_value=100
        ):
            paths, _ = task_service.generate_final_videos(
                task_id="task-thumb-off",
                params=params,
                downloaded_videos=["clip.mp4"],
                audio_file="audio.mp3",
                subtitle_path="subtitle.srt",
            )

        self.assertEqual(len(paths), 2)
        thumb_mock.assert_not_called()

    def test_enabled_generates_one_thumbnail_per_final_video(self):
        params = _params(video_count=2)
        with patch(
            "app.services.task.config",
            _thumbnail_config_get(True),
        ), patch.object(
            task_service.sm.state, "update_task"
        ), patch.object(
            task_service.video, "combine_videos"
        ), patch.object(
            task_service.video, "generate_video"
        ), patch.object(
            task_service.thumbnail_service, "generate_thumbnail"
        ) as thumb_mock, patch(
            "app.services.task.utils.task_dir",
            side_effect=lambda task_id: rf"C:\tasks\{task_id}",
        ), patch(
            "app.services.task.path.join",
            side_effect=os.path.join,
        ), patch(
            "app.services.task.path.isfile", return_value=True
        ), patch(
            "app.services.task.path.getsize", return_value=100
        ):
            task_service.generate_final_videos(
                task_id="task-thumb-on",
                params=params,
                downloaded_videos=["clip.mp4"],
                audio_file="audio.mp3",
                subtitle_path="subtitle.srt",
            )

        self.assertEqual(thumb_mock.call_count, 2)
        thumb_mock.assert_any_call(
            "C:\\tasks\\task-thumb-on\\final-1.mp4",
            title="SSD Upgrade Guide",
            font_name="MissingFont.ttc",
        )
        thumb_mock.assert_any_call(
            "C:\\tasks\\task-thumb-on\\final-2.mp4",
            title="SSD Upgrade Guide",
            font_name="MissingFont.ttc",
        )


if __name__ == "__main__":
    unittest.main()
