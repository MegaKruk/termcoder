"""One-time LiteLLM configuration.

Called once at startup. ``drop_params`` lets LiteLLM silently drop parameters a
given provider does not support, which keeps a single code path working across
OpenAI, Anthropic and local Ollama models.
"""

from __future__ import annotations

import logging
import warnings


def configure_litellm() -> None:
    """Apply process-wide LiteLLM settings and quiet its default logging."""
    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    _silence_litellm_serializer_warnings()


def _silence_litellm_serializer_warnings() -> None:
    """Hide the benign pydantic serializer warnings LiteLLM emits.

    When a model is served through the OpenAI Responses API, LiteLLM's own
    logging reshapes the usage object and pydantic then prints a
    "Pydantic serializer warnings" UserWarning for every chunk. It is harmless
    but it floods the terminal and can splice into streamed output. The filter
    is scoped to pydantic's serializer so real validation errors, which are
    raised as exceptions rather than warnings, are unaffected.
    """
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pydantic.main",
    )
