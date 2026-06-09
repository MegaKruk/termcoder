"""Session metadata.

Metadata is stored in a small JSON sidecar next to each chat transcript so the
session list can be shown without reading every message line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SessionMeta:
    """Lightweight metadata describing one chat session."""

    id: str
    title: str
    created_at: str
    updated_at: str
    model: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMeta":
        return cls(
            id=data["id"],
            title=data.get("title", "untitled"),
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
            model=data.get("model", "unknown"),
        )
