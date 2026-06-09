"""Non-interactive approvers.

These are used by tests and any future headless mode. They contain no
user-interface code.
"""

from __future__ import annotations

from .types import ApprovalOutcome, ApprovalRequest, Decision


class AutoApprover:
    """Approve every request without prompting. For tests and headless runs."""

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        return ApprovalOutcome(Decision.APPROVE)


class RejectingApprover:
    """Reject every request. Useful for verifying rejection handling."""

    def __init__(self, feedback: str | None = None):
        self._feedback = feedback

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        return ApprovalOutcome(Decision.REJECT, feedback=self._feedback)
