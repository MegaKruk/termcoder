"""Approval types and helpers (user-interface free)."""

from .auto import AutoApprover, RejectingApprover
from .diffing import make_unified_diff
from .types import ApprovalOutcome, ApprovalRequest, Approver, Decision

__all__ = [
    "ApprovalOutcome",
    "ApprovalRequest",
    "Approver",
    "Decision",
    "AutoApprover",
    "RejectingApprover",
    "make_unified_diff",
]
