import os
import logging
import backoff
from typing import (
    Dict,
    Sequence,
    Optional,
    List,
    Any,
    TypeVar,
    Callable,
    Generator,
    Union,
    Literal,
)
import re

from adalflow.utils.lazy_import import safe_import, OptionalPackages
from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    CompletionUsage,
    GeneratorOutput,
)

deepseek = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
    Image,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from adalflow.components.model_client.utils import parse_embedding_response

# Using OpenAI SDK to access DeepSeek API

log = logging.getLogger(__name__)
T = TypeVar("T")


# completion parsing functions and you can combine them into one singple chat completion parser
def parse_deepseek_response(completion: ChatCompletion) -> str:
    """Parse the response of the DeepSeek API."""
    return completion.choices[0].message.content

def parse_stream_response(completion: ChatCompletionChunk) -> str:
    r"""Parse the response of the stream API."""
    return completion.choices[0].delta.content

def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """Handle the streaming response from DeepSeek API."""
    try:
        for completion in generator:
            log.debug(f"Raw chunk completion: {completion}")
            parsed_content = parse_stream_response(completion)
            yield parsed_content
    except Exception as e:
        log.error(f"Error in streaming response: {e}")
        raise

# A simple heuristic to estimate token count for estimating number of tokens in a Streaming response
def estimate_token_count(text: str) -> int:
    """
    Estimate the token count of a given text.

    Args:
        text (str): The text to estimate token count for.

    Returns:
        int: Estimated token count.
    """
    # Split the text into tokens using spaces as a simple heuristic
    tokens = text.split()

    # Return the number of tokens
    return len(tokens)



class DeepSeekClient(ModelClient):
    """
    A component wrapper for the DeepSeek API client.

    The DeepSeek API uses an API format compatible with OpenAI. By modifying the configuration,
    you can use the OpenAI SDK or software compatible with the OpenAI API to access the DeepSeek API.

    The documentation follows the documentation in "https://api-docs.deepseek.com/guides/reasoning_model" 

    Args:
        api_key (Optional[str], optional): DeepSeek API key. Defaults to None.
        response_parser (Callable[[Dict], Any], optional): A function to parse API responses. Defaults to `parse_deepseek_response`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Dict], Any] = None,
        input_type: Literal["text", "messages"] = "text",
    ):
        """Initialize the DeepSeek API client."""
        super().__init__()
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise ValueError("DeepSeek API key must be provided or set in the environment.")
        self.chat_completion_parser = (
            chat_completion_parser or parse_deepseek_response
        )
        self.sync_client = self.init_sync_client()
        self._input_type = input_type
        self._api_kwargs = {} # add api kwargs when the DeepSeek Client is called


    def init_sync_client(self):
        """Initialize the synchronous DeepSeek API client."""
        return deepseek.OpenAI(api_key=self._api_key, base_url="https://api.deepseek.com")

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """Parse the completion, and put it into the raw_response."""
        log.debug(f"completion: {completion}, parser: {self.chat_completion_parser}")
        try:
            data = self.chat_completion_parser(completion)
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=data, usage=usage
            )
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """
        Track token usage for both non-streaming and streaming completions.
        """
        if isinstance(completion, ChatCompletion):
            usage: CompletionUsage = CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            return usage
        elif isinstance(completion, Stream):
            # Streaming response
            completion_tokens = 0
            prompt_tokens = 0

            for message in self._api_kwargs.get("messages", []):
                # add to prompt_tokens if the message role is 'system' which contains system prompt 
                if message.get("role") == "system": 
                    content = message.get("content", '') 
                    prompt_tokens += estimate_token_count(content)
                    break 

            for chunk in completion:
                if hasattr(chunk.choices[0].delta, "content"):
                    # Estimate token count from streamed content
                    completion_tokens += estimate_token_count(parse_stream_response(chunk))
            # Since prompt tokens are known in advance, retrieve from model kwargs or a known value
            total_tokens = prompt_tokens + completion_tokens

            usage: CompletionUsage = CompletionUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )
            return usage

        else:
            raise ValueError(f"Unsupported completion type: {type(completion)}")
        
    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure Adalflow components can understand.

        Should be called in ``Embedder``.
        """
        try:
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)
    
    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert inputs to DeepSeek API-specific format.

        Args:
            input: The input text or messages to process.
            model_kwargs: Additional parameters.
            model_type: The type of model (e.g., LLM or Embedder).

        Returns:
            Dict: API-specific kwargs for the model call.
        """
        api_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            api_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            # Construct the messages list
            system_prompt = api_kwargs.pop(
                "system_prompt", "You are a helpful assistant."
            )  # Default system prompt
            messages = [{"role": "system", "content": system_prompt}]
            # Add the user input as the final message
            if isinstance(input, str):
                messages.append({"role": "user", "content": input})
            else:
                raise TypeError("Input must be a string for LLM models.")
            api_kwargs["messages"] = messages
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return api_kwargs


    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs.  Support streaming call.
        """
        log.info(f"api_kwargs: {api_kwargs}")
        self._api_kwargs = api_kwargs
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                self.chat_completion_parser = handle_streaming_response
                return self.sync_client.chat.completions.create(**api_kwargs)
            return self.sync_client.chat.completions.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """Create a DeepSeekClient instance from a dictionary."""
        obj = super().from_dict(data)
        obj.sync_client = obj.init_sync_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert the component to a dictionary."""
        exclude = ["sync_client"]
        output = super().to_dict(exclude=exclude)
        return output


# Example usage:
# if __name__ == "__main__":
#     from adalflow.core import Generator
#     from adalflow.utils import setup_env, get_logger

#     log = get_logger(level="DEBUG")

#     # setup_env()
#     prompt_kwargs = {"input_str": "What is the meaning of life?"}

#     gen = Generator(
#         model_client=DeepSeekClient(),
#         model_kwargs={"model": "deepseek-reasoner", "stream": False},
#     )
    
#     gen_response = gen(prompt_kwargs)
#     print(f"gen_response: {gen_response}")

#     for genout in gen_response.data:
#         print(f"genout: {genout}")


