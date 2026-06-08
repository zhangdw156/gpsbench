import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.llm_client import LLMClient


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.timeout = None

    def list(self, timeout=None):
        self.calls += 1
        self.timeout = timeout
        return self.response


class FakeOpenAIClient:
    def __init__(self, response=None):
        self.models = FakeModels(response or SimpleNamespace(data=[]))


class LLMClientContextLimitTests(unittest.TestCase):
    def test_default_max_tokens_is_half_context_from_models_endpoint(self):
        fake_client = FakeOpenAIClient(
            SimpleNamespace(data=[{"id": "served-model", "max_model_len": 32768}])
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}, clear=False):
            with patch("evaluation.llm_client.openai.OpenAI", return_value=fake_client):
                client = LLMClient(provider="openai", model="served-model", max_tokens=None, timeout=17)

        self.assertEqual(client.model_context_length, 32768)
        self.assertEqual(client.max_tokens, 16384)
        self.assertEqual(client.max_tokens_source, "half_context_from_v1_models:32768")
        self.assertEqual(fake_client.models.calls, 1)
        self.assertEqual(fake_client.models.timeout, 17)

    def test_context_length_can_be_read_from_nested_camel_case_metadata(self):
        fake_client = FakeOpenAIClient(
            SimpleNamespace(data=[{"id": "served-model", "metadata": {"maxModelLen": "65,536"}}])
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}, clear=False):
            with patch("evaluation.llm_client.openai.OpenAI", return_value=fake_client):
                client = LLMClient(provider="openai", model="served-model", max_tokens=None)

        self.assertEqual(client.model_context_length, 65536)
        self.assertEqual(client.max_tokens, 32768)

    def test_explicit_max_tokens_skips_models_endpoint_lookup(self):
        fake_client = FakeOpenAIClient(
            SimpleNamespace(data=[{"id": "served-model", "max_model_len": 32768}])
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}, clear=False):
            with patch("evaluation.llm_client.openai.OpenAI", return_value=fake_client):
                client = LLMClient(provider="openai", model="served-model", max_tokens=2048)

        self.assertEqual(client.max_tokens, 2048)
        self.assertEqual(client.max_tokens_source, "explicit")
        self.assertEqual(fake_client.models.calls, 0)

    def test_reasoning_batch_requests_include_output_cap(self):
        fake_client = FakeOpenAIClient()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}, clear=False):
            with patch("evaluation.llm_client.openai.OpenAI", return_value=fake_client):
                client = LLMClient(provider="openai", model="gpt-5.1", max_tokens=4096)

        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "batch.jsonl"
            client.create_batch_file(["prompt"], system_prompt="system", output_file=str(output_file))
            task = json.loads(output_file.read_text(encoding="utf-8").strip())

        self.assertEqual(task["url"], "/v1/responses")
        self.assertEqual(task["body"]["max_output_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
