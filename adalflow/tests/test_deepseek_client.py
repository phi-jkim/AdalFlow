import unittest
from unittest.mock import patch, Mock
import os

from openai import Stream 
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk 
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.components.model_client.deepseek_client import DeepSeekClient, estimate_token_count
from unittest.mock import AsyncMock

def getenv_side_effect(key):
    env_vars = {"DEEPSEEK_API_KEY": "fake_api_key"}
    return env_vars.get(key, None)

# Mock the Stream object
class MockStream(Stream):
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)

class TestDeepSeekClient(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekClient(api_key="fake_api_key")

        self.mock_response = ChatCompletion(
            id="cmpl-3Q8Z5J9Z1Z5z5",
            created=1635820005,
            object="chat.completion",
            model="gpt-4o",
            choices=[
                {
                    "message": {"content": "Hello, world!", "role": "assistant"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            usage=CompletionUsage(
                completion_tokens=10, prompt_tokens=20, total_tokens=30
            ),
        )

        # Correct mock_stream data to include all required fields
        self.mock_stream = [
            ChatCompletionChunk(
                id="chunk-1",
                object="chat.completion.chunk",
                created=1635820005,
                model="gpt-4o",
                choices=[
                    {"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}
                ],
            ),
            ChatCompletionChunk(
                id="chunk-2",
                object="chat.completion.chunk",
                created=1635820005,
                model="gpt-4o",
                choices=[
                    {"delta": {"content": ", world!"}, "index": 0, "finish_reason": "stop"}
                ],
            ),
        ]

        self.api_kwargs_non_streaming = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, DeepSeek!"},
            ],
            "model": "deepseek-reasoner",
            "stream": False,
        }

        self.api_kwargs_streaming = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, DeepSeek!"},
            ],
            "model": "deepseek-reasoner",
            "stream": True,
        }

    def parse_stream_response(self, completion: ChatCompletionChunk) -> str:
        r"""Parse the response of the stream API."""
        return completion.choices[0].delta.content

    # helper function to parse the streaming response 
    def handle_streaming_response(self, generator: Stream[ChatCompletionChunk]):
        """Handle the streaming response from DeepSeek API."""
        try:
            for completion in generator:
                parsed_content = self.parse_stream_response(completion)
                yield parsed_content
        except Exception as e:
            raise

    @patch("adalflow.components.model_client.deepseek_client.deepseek.OpenAI")
    def test_call_non_streaming(self, MockSyncOpenAI):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
        mock_sync_client.chat.completions.create.return_value = self.mock_response

        self.client.sync_client = mock_sync_client

        result = self.client.call(
            api_kwargs=self.api_kwargs_non_streaming, model_type=ModelType.LLM
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            **self.api_kwargs_non_streaming
        )
        self.assertEqual(result, self.mock_response)

    @patch("adalflow.components.model_client.deepseek_client.deepseek.OpenAI")
    def test_call_streaming(self, MockSyncOpenAI):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
        mock_sync_client.chat.completions.create.return_value = iter(self.mock_stream)

        self.client.sync_client = mock_sync_client

        result = self.client.call(
            api_kwargs=self.api_kwargs_streaming, model_type=ModelType.LLM
        )

        mock_sync_client.chat.completions.create.assert_called_once_with(
            **self.api_kwargs_streaming
        )

        # Collect the streamed chunks
        streamed_content = "".join(
            [chunk.choices[0].delta.content for chunk in result]
        )
        self.assertEqual(streamed_content, "Hello, world!")

    def test_track_completion_usage_streaming(self):

        mock_stream = MockStream(self.mock_stream)

        self.client._api_kwargs = self.api_kwargs_streaming

        usage = self.client.track_completion_usage(mock_stream)

        # Check the token estimation logic
        expected_prompt_tokens = estimate_token_count(
            "You are a helpful assistant."
        )
        expected_completion_tokens = estimate_token_count("Hello") + estimate_token_count(", world!")
        self.assertEqual(usage.prompt_tokens, expected_prompt_tokens)
        self.assertEqual(usage.completion_tokens, expected_completion_tokens)
        self.assertEqual(
            usage.total_tokens, expected_prompt_tokens + expected_completion_tokens
        )

    def test_track_completion_usage_non_streaming(self):
        usage = self.client.track_completion_usage(self.mock_response)

        self.assertEqual(usage.prompt_tokens, 20)
        self.assertEqual(usage.completion_tokens, 10)
        self.assertEqual(usage.total_tokens, 30)

    @patch("adalflow.components.model_client.deepseek_client.deepseek.OpenAI")
    def test_parse_chat_completion_streaming(self, MockSyncOpenAI):
        mock_sync_client = Mock()
        MockSyncOpenAI.return_value = mock_sync_client
        mock_sync_client.chat.completions.create.return_value = iter(self.mock_stream)

        self.client.sync_client = mock_sync_client

        completion = iter(self.mock_stream)
        result = "".join(self.handle_streaming_response(completion))

        self.assertEqual(result, "Hello, world!")

if __name__ == "__main__":
    unittest.main()
