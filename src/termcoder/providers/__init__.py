"""Language model provider layer.

Importing this package does not import LiteLLM. The heavy import happens lazily
inside :class:`LLMClient` and :func:`configure_litellm`, which keeps the rest
of the package (and the test suite) free of that dependency.
"""

from .llm_client import CompletionResult, LLMClient
from .setup import configure_litellm

__all__ = ["LLMClient", "CompletionResult", "configure_litellm"]
