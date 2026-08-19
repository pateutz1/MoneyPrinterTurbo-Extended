import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoParams
from app.services import llm
from app.services import task as task_service

LLM_SECRET = "secret-llm-key-do-not-leak"
GLOBAL_TERMS = ["global one", "global two", "global three", "global four", "global five"]


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
        video_subject="SSD upgrades",
        video_script="",
        video_terms=None,
        video_aspect="9:16",
        video_concat_mode="random",
        video_transition_mode="None",
        video_clip_duration=5,
        video_count=1,
        video_source="pexels",
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
        font_name="MicrosoftYaHeiBold.ttc",
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


class TestSentenceKeywordParser(unittest.TestCase):
    SCRIPT_THREE = (
        "SSDs make boot times faster. "
        "They also reduce load times in games. "
        "Many users upgrade older laptops."
    )

    def test_valid_json_one_keyword_per_sentence(self):
        parsed = llm.parse_sentence_keywords_response(
            '["fast boot", "game loading", "laptop upgrade"]',
            expected_count=3,
        )
        self.assertEqual(parsed, ["fast boot", "game loading", "laptop upgrade"])

    def test_json_inside_markdown_fence(self):
        parsed = llm.parse_sentence_keywords_response(
            'Here you go:\n```json\n["fast boot", "game loading", "laptop upgrade"]\n```',
            expected_count=3,
        )
        self.assertEqual(parsed, ["fast boot", "game loading", "laptop upgrade"])

    def test_duplicate_keyword_entry_fails(self):
        with self.assertRaises(ValueError) as ctx:
            llm.parse_sentence_keywords_response(
                '["SSD storage", "ssd storage", "laptop upgrade"]',
                expected_count=3,
            )
        self.assertEqual(str(ctx.exception), "duplicate keyword entry")

    def test_duplicate_keyword_output_fails_closed(self):
        with patch.object(
            llm,
            "_generate_response",
            return_value='["SSD storage", "ssd storage", "laptop upgrade"]',
        ):
            keywords, reason = llm.generate_sentence_keywords(self.SCRIPT_THREE)
        self.assertIsNone(keywords)
        self.assertEqual(reason, "duplicate keyword entry")

    def test_blank_keyword_fails(self):
        with self.assertRaises(ValueError) as ctx:
            llm.parse_sentence_keywords_response(
                '["SSD storage", "   ", "laptop upgrade"]',
                expected_count=3,
            )
        self.assertEqual(str(ctx.exception), "empty keyword entry")

    def test_malformed_json_fails(self):
        with patch.object(llm, "_generate_response", return_value="not-json-at-all"):
            keywords, reason = llm.generate_sentence_keywords("One sentence here.")
        self.assertIsNone(keywords)
        self.assertEqual(reason, "malformed json")

    def test_empty_output_fails(self):
        with patch.object(llm, "_generate_response", return_value="[]"):
            keywords, reason = llm.generate_sentence_keywords("One sentence here.")
        self.assertIsNone(keywords)
        self.assertEqual(reason, "incomplete keyword count")

    def test_provider_error_fails(self):
        with patch.object(llm, "_generate_response", return_value="Error: provider down"):
            keywords, reason = llm.generate_sentence_keywords("One sentence here.")
        self.assertIsNone(keywords)
        self.assertEqual(reason, "provider error")

    def test_more_than_five_sentences_bounded(self):
        script = ". ".join(f"Sentence number {index}" for index in range(1, 9)) + "."
        sentences = llm.split_script_sentences(script)
        self.assertEqual(len(sentences), 5)

        response = '["one", "two", "three", "four", "five"]'
        with patch.object(llm, "_generate_response", return_value=response):
            keywords, reason = llm.generate_sentence_keywords(script)
        self.assertIsNone(reason)
        self.assertEqual(keywords, ["one", "two", "three", "four", "five"])

    def test_excessive_keyword_count_fails(self):
        with patch.object(
            llm,
            "_generate_response",
            return_value='["a", "b", "c", "d", "e", "f"]',
        ):
            keywords, reason = llm.generate_sentence_keywords(self.SCRIPT_THREE)
        self.assertIsNone(keywords)
        self.assertEqual(reason, "excessive keyword count")


def _config_get(enabled):
    mock_config = MagicMock()

    def _get(key, default=None):
        if key == "enable_sentence_keywords":
            return enabled
        return default

    mock_config.app.get = _get
    return mock_config


class TestOllamaThinkingDisabled(unittest.TestCase):
    SCRIPT_THREE = (
        "Solid state drives make boot times faster. "
        "They also reduce loading times in modern games. "
        "Many users upgrade older laptops with SSD storage."
    )

    def _ollama_config_get(self, key, default=None):
        values = {
            "llm_provider": "ollama",
            "ollama_model_name": "qwen3:4b",
            "ollama_base_url": "http://localhost:11434/v1",
        }
        return values.get(key, default)

    def _mock_chat_completion(self, content):
        fake_choice = MagicMock()
        fake_choice.message.content = content
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        real_isinstance = isinstance

        def isinstance_patch(obj, cls):
            if cls is llm.ChatCompletion and obj is fake_response:
                return True
            return real_isinstance(obj, cls)

        return mock_client, isinstance_patch

    def test_sentence_keywords_requests_disable_thinking(self):
        captured = []

        def fake_generate_response(
            prompt, disable_thinking=False, expected_keyword_count=None
        ):
            captured.append((disable_thinking, expected_keyword_count))
            return '["fast boot", "game loading", "laptop upgrade"]'

        with patch.object(llm, "_generate_response", side_effect=fake_generate_response):
            keywords, reason = llm.generate_sentence_keywords(self.SCRIPT_THREE)

        self.assertIsNone(reason)
        self.assertEqual(keywords, ["fast boot", "game loading", "laptop upgrade"])
        self.assertEqual(captured, [(True, 3)])

    def test_generate_terms_does_not_disable_thinking(self):
        captured = []

        def fake_generate_response(
            prompt, disable_thinking=False, expected_keyword_count=None
        ):
            captured.append((disable_thinking, expected_keyword_count))
            return '["one", "two", "three", "four", "five"]'

        with patch.object(llm, "_generate_response", side_effect=fake_generate_response):
            terms = llm.generate_terms("SSD upgrades", self.SCRIPT_THREE, amount=5)

        self.assertEqual(terms, ["one", "two", "three", "four", "five"])
        self.assertTrue(captured)
        self.assertTrue(all(flag is False and count is None for flag, count in captured))

    def test_generate_script_does_not_use_sentence_keyword_request_fields(self):
        captured = []

        def fake_generate_response(
            prompt, disable_thinking=False, expected_keyword_count=None
        ):
            captured.append((disable_thinking, expected_keyword_count))
            return "Solid state drives boot quickly."

        with patch.object(llm, "_generate_response", side_effect=fake_generate_response):
            script = llm.generate_script(
                video_subject="SSD upgrades",
                language="en",
                paragraph_number=1,
            )

        self.assertTrue(script)
        self.assertTrue(captured)
        self.assertTrue(all(flag is False and count is None for flag, count in captured))

    def test_ollama_sentence_keyword_request_fields(self):
        expected_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentence_keywords",
                "strict": True,
                "schema": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
            },
        }
        mock_client, isinstance_patch = self._mock_chat_completion(
            '["fast boot", "game loading", "laptop upgrade"]'
        )

        with patch.object(llm.config, "app") as mock_app, patch.object(
            llm, "OpenAI", return_value=mock_client
        ), patch("app.services.llm.isinstance", side_effect=isinstance_patch):
            mock_app.get = self._ollama_config_get
            result = llm._generate_response(
                "prompt",
                disable_thinking=True,
                expected_keyword_count=3,
            )

        self.assertEqual(
            result, '["fast boot", "game loading", "laptop upgrade"]'
        )
        captured_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            captured_kwargs.get("extra_body"), {"reasoning_effort": "none"}
        )
        self.assertEqual(captured_kwargs.get("max_tokens"), 128)
        self.assertEqual(captured_kwargs.get("response_format"), expected_format)

    def test_ollama_normal_request_omits_sentence_keyword_fields(self):
        mock_client, isinstance_patch = self._mock_chat_completion(
            '["one", "two", "three", "four", "five"]'
        )

        with patch.object(llm.config, "app") as mock_app, patch.object(
            llm, "OpenAI", return_value=mock_client
        ), patch("app.services.llm.isinstance", side_effect=isinstance_patch):
            mock_app.get = self._ollama_config_get
            llm._generate_response("prompt", disable_thinking=False)

        captured_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", captured_kwargs)
        self.assertNotIn("max_tokens", captured_kwargs)
        self.assertNotIn("response_format", captured_kwargs)

    def test_non_ollama_provider_omits_ollama_controls(self):
        mock_client, isinstance_patch = self._mock_chat_completion(
            '["fast boot", "game loading", "laptop upgrade"]'
        )

        def openai_config_get(key, default=None):
            values = {
                "llm_provider": "openai",
                "openai_api_key": "test-key",
                "openai_model_name": "gpt-4o-mini",
                "openai_base_url": "https://api.openai.com/v1",
            }
            return values.get(key, default)

        with patch.object(llm.config, "app") as mock_app, patch.object(
            llm, "OpenAI", return_value=mock_client
        ), patch("app.services.llm.isinstance", side_effect=isinstance_patch):
            mock_app.get = openai_config_get
            llm._generate_response(
                "prompt",
                disable_thinking=True,
                expected_keyword_count=3,
            )

        captured_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", captured_kwargs)
        self.assertNotIn("max_tokens", captured_kwargs)
        self.assertNotIn("response_format", captured_kwargs)


class TestTaskGenerateTerms(unittest.TestCase):
    SCRIPT = (
        "SSDs make boot times faster. "
        "They also reduce load times in games."
    )

    def test_feature_disabled_uses_global_terms_only(self):
        params = _params()
        with patch("app.services.task.config", _config_get(False)), patch.object(
            llm, "generate_sentence_keywords"
        ) as sentence_mock, patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ) as global_mock:
            result = task_service.generate_terms("task-disabled", params, self.SCRIPT)

        self.assertEqual(result, GLOBAL_TERMS)
        sentence_mock.assert_not_called()
        global_mock.assert_called_once_with(
            video_subject="SSD upgrades",
            video_script=self.SCRIPT,
            amount=5,
        )

    def test_valid_sentence_keywords_used(self):
        params = _params()
        sentence_terms = ["fast boot", "game loading"]
        with patch("app.services.task.config", _config_get(True)), patch.object(
            llm, "generate_sentence_keywords", return_value=(sentence_terms, None)
        ), patch.object(llm, "generate_terms") as global_mock:
            result = task_service.generate_terms("task-valid", params, self.SCRIPT)

        self.assertEqual(result, sentence_terms)
        global_mock.assert_not_called()

    def test_malformed_json_falls_back_to_global_terms(self):
        params = _params()
        with patch("app.services.task.config", _config_get(True)), patch.object(
            llm, "generate_sentence_keywords", return_value=(None, "malformed json")
        ), patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ) as global_mock:
            logs = _LogCapture()
            with logs:
                result = task_service.generate_terms("task-malformed", params, self.SCRIPT)

        self.assertEqual(result, GLOBAL_TERMS)
        global_mock.assert_called_once()
        self.assertIn("sentence keywords fallback: malformed json", logs.text())

    def test_empty_output_falls_back_to_global_terms(self):
        params = _params()
        with patch("app.services.task.config", _config_get(True)), patch.object(
            llm, "generate_sentence_keywords", return_value=(None, "empty result")
        ), patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ):
            logs = _LogCapture()
            with logs:
                result = task_service.generate_terms("task-empty", params, self.SCRIPT)

        self.assertEqual(result, GLOBAL_TERMS)
        self.assertIn("sentence keywords fallback: empty result", logs.text())

    def test_provider_exception_falls_back_to_global_terms(self):
        params = _params()
        with patch("app.services.task.config", _config_get(True)), patch.object(
            llm, "generate_sentence_keywords", return_value=(None, "provider error")
        ), patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ):
            logs = _LogCapture()
            with logs:
                result = task_service.generate_terms("task-provider", params, self.SCRIPT)

        self.assertEqual(result, GLOBAL_TERMS)
        self.assertIn("sentence keywords fallback: provider error", logs.text())

    def test_duplicate_output_falls_back_to_global_terms(self):
        params = _params()
        with patch("app.services.task.config", _config_get(True)), patch.object(
            llm,
            "_generate_response",
            return_value='["fast boot", "fast boot", "game loading"]',
        ), patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ) as global_mock:
            logs = _LogCapture()
            with logs:
                result = task_service.generate_terms(
                    "task-duplicate", params, self.SCRIPT + " Many users upgrade laptops."
                )

        self.assertEqual(result, GLOBAL_TERMS)
        global_mock.assert_called_once()
        self.assertIn("sentence keywords fallback: duplicate keyword entry", logs.text())

    def test_unexpected_exception_falls_back_to_global_terms(self):
        params = _params()
        with patch("app.services.task.config", _config_get(True)), patch.object(
            llm,
            "generate_sentence_keywords",
            side_effect=RuntimeError(LLM_SECRET),
        ), patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ) as global_mock:
            logs = _LogCapture()
            with logs:
                result = task_service.generate_terms("task-runtime", params, self.SCRIPT)

        self.assertEqual(result, GLOBAL_TERMS)
        global_mock.assert_called_once()
        blob = logs.text()
        self.assertIn("sentence keywords fallback: unexpected error", blob)
        self.assertNotIn(LLM_SECRET, blob)
        self.assertNotIn("RuntimeError", blob)

    def test_generate_response_exception_is_safe_when_called_directly(self):
        with patch.object(
            llm,
            "_generate_response",
            side_effect=RuntimeError(LLM_SECRET),
        ):
            logs = _LogCapture()
            with logs:
                keywords, reason = llm.generate_sentence_keywords(self.SCRIPT)

        self.assertIsNone(keywords)
        self.assertEqual(reason, "provider error")
        blob = logs.text()
        self.assertNotIn(LLM_SECRET, blob)
        self.assertNotIn("RuntimeError", blob)

    def test_no_secret_in_logs_or_exceptions(self):
        params = _params()
        secret_response = (
            f'Authorization: Bearer {LLM_SECRET}\n'
            '```json\n["fast boot", "game loading"]\n```'
        )
        logs = _LogCapture()
        with logs, patch("app.services.task.config", _config_get(True)), patch.object(
            llm, "_generate_response", return_value=secret_response
        ), patch.object(
            llm, "generate_terms", return_value=GLOBAL_TERMS
        ):
            keywords, reason = llm.generate_sentence_keywords(self.SCRIPT)
            if keywords:
                task_service.generate_terms("task-secret", params, self.SCRIPT)

        blob = logs.text()
        self.assertNotIn(LLM_SECRET, blob)
        self.assertNotIn("Authorization: Bearer", blob)


if __name__ == "__main__":
    unittest.main()
