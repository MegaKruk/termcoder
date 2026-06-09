"""Chat message helpers.

Messages use the OpenAI chat format because LiteLLM normalizes every provider
to it. Keeping messages as plain dictionaries means they can be written to disk
and replayed into the model without any custom serialization.
"""

from __future__ import annotations

from typing import Any


def system_message(content: str) -> dict:
    """Build a system message."""
    return {"role": "system", "content": content}


def user_message(content: str) -> dict:
    """Build a user message."""
    return {"role": "user", "content": content}


def tool_message(tool_call_id: str, content: str) -> dict:
    """Build a tool-result message tied to a specific tool call."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def assistant_message_to_dict(message: Any) -> dict:
    """Convert a LiteLLM assistant message object into a storable dict.

    Preserves any tool calls in OpenAI format so the message can be replayed in
    a later request, which is required for multi-step tool use.
    """
    data: dict = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return data
