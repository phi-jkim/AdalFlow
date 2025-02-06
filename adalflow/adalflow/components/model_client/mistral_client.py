from typing import Optional, Any, Callable, Literal
from openai.types import Completion
from adalflow.components.model_client.openai_client import OpenAIClient

BASE_URL = "https://api.mistral.ai/v1"

__doc__ = r"""
A minimal Mistral client that inherits from :class:`OpenAIClient`.

This client is designed to work with Mistral’s API by setting:
  - The API base URL to ``https://api.mistral.ai/v1``.
  - The API key is fetched from the environment variable ``MISTRAL_API_KEY`` if not provided.
  - The input format is supported as either ``"text"`` or ``"messages"``.
  - The AdalFlow Generator is expected to supply additional model parameters (such as model name, temperature, and max_tokens)
    in a single configuration point.

**Example usage with the AdalFlow Generator:**

.. code-block:: python

    import os
    from adalflow.core import Generator
    from adalflow.components.model_client.mistral_client import MistralClient
    from adalflow.utils import setup_env, get_logger

    # Set up logging and load environment variables.
    get_logger(enable_file=False)
    setup_env()

    # Create a MistralClient instance. The API key will be read from the environment if not provided.
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"), input_type="messages")

    # Initialize the Generator with the MistralClient and desired model parameters.
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "mistral-large-latest",
            "temperature": 0.7,
            "max_tokens": 2000,
        }
    )

    # Define the prompt to send to the model.
    prompt_kwargs = {"input_str": "Explain the concept of machine learning."}

    # Generate a response using the configured generator.
    response = generator(prompt_kwargs)

    # Check for errors and print the output.
    if response.error:
        print(f"[Mistral] Generator Error: {response.error}")
    else:
        print(f"[Mistral] Response: {response.data}")
"""


class MistralClient(OpenAIClient):
    r"""
    A minimal Mistral client that inherits from :class:`OpenAIClient`.

    This client customizes the following defaults:
      - ``base_url``: Set to ``https://api.mistral.ai/v1``.
      - ``env_api_key_name``: Uses ``MISTRAL_API_KEY`` to retrieve the API key from the environment.
      - ``input_type``: Defaults to ``"messages"``, supporting both text and message-based inputs.

    The AdalFlow Generator is expected to provide specific model parameters (such as model name,
    temperature, and max_tokens) in a centralized configuration.

    See the module-level documentation for an example of how to use :class:`MistralClient`
    with the AdalFlow Generator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        input_type: Literal["text", "messages"] = "text",
        env_api_key_name: str = "MISTRAL_API_KEY",
        chat_completion_parser: Callable[[Completion], Any] = None,
    ):
        """
        Initialize a MistralClient instance.

        :param api_key: Mistral API key. If None, reads from the environment variable ``MISTRAL_API_KEY``.
        :param base_url: URL for Mistral’s endpoint (default: ``https://api.mistral.ai/v1``).
        :param input_type: Input format, either ``"text"`` or ``"messages"``.
        :param env_api_key_name: Name of the environment variable to use for the Mistral API key (default: ``MISTRAL_API_KEY``).
        :param chat_completion_parser: Optional function to parse responses from Mistral's API.
        """
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_api_key_name=env_api_key_name,
        )


if __name__ == "__main__":
    import os
    from adalflow.core import Generator
    from adalflow.utils import setup_env, get_logger

    # Set up logging and load environment variables.
    get_logger(enable_file=False)
    setup_env()

    # Instantiate the MistralClient; the API key will be obtained from the environment if not explicitly provided.
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"), input_type="messages")

    # Create the Generator using the MistralClient and specify model parameters.
    generator = Generator(
        model_client=client,
        model_kwargs={
            "model": "mistral-large-latest",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    )

    # Define the prompt to be processed.
    prompt_kwargs = {"input_str": "Explain the concept of machine learning."}

    # Generate and output the response.
    response = generator(prompt_kwargs)

    if response.error:
        print(f"[Mistral] Generator Error: {response.error}")
    else:
        print(f"[Mistral] Response: {response.data}")
