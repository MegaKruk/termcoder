"""One-time LiteLLM configuration.

Called once at startup. ``drop_params`` lets LiteLLM silently drop parameters a
given provider does not support, which keeps a single code path working across
OpenAI, Anthropic and local Ollama models.
"""

from __future__ import annotations

import logging


def configure_litellm() -> None:
    """Apply process-wide LiteLLM settings and quiet its default logging."""
    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
