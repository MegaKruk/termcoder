"""Approval domain types.

These types are deliberately free of any user-interface code so that the
agent loop and tools can depend on them without pulling in a terminal stack.
Concrete approvers (a console approver, an auto-approver for tests) implement
the :class:`Approver` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Decision(Enum):
    """The possible outcomes of an approval prompt."""

    APPROVE = "approve"
    APPROVE_FOR_SESSION = "approve_for_session"
    REJECT = "reject"


@dataclass
class ApprovalRequest:
    """A request to perform a non-read-only action that needs user consent."""

    tool_name: str
    summary: str
    detail: str | None = None
    detail_kind: str = "text"  # one of: "text", "diff", "command"
    destructive: bool = False
    note: str | None = None


@dataclass
class ApprovalOutcome:
    """The user's response to an approval request."""

    decision: Decision
    feedback: str | None = None

    @property
    def approved(self) -> bool:
        """True when the action may proceed."""
        return self.decision in (Decision.APPROVE, Decision.APPROVE_FOR_SESSION)


class Approver(Protocol):
    """Anything that can decide whether an action may proceed."""

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        """Return the user's decision for the given request."""
        ...
