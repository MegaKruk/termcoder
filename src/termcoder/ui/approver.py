"""Interactive approval prompt.

Implements the three-way choice used by GitHub Copilot: approve once, approve
this tool for the rest of the session, or reject with optional feedback. Tools
approved for the session are remembered here so the user is not asked again for
the same tool.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession

from ..approval.types import ApprovalOutcome, ApprovalRequest, Decision
from .renderer import Renderer

_PROMPT = "Approve? [y] once  [a] allow this tool for the session  [n] reject: "


class ConsoleApprover:
    """Ask the user to approve actions, remembering session-wide approvals."""

    def __init__(self, renderer: Renderer, prompt_session: PromptSession | None = None):
        self._renderer = renderer
        self._prompt = prompt_session or PromptSession()
        self._approved_tools: set[str] = set()

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        if request.tool_name in self._approved_tools:
            return ApprovalOutcome(Decision.APPROVE_FOR_SESSION)

        self._renderer.render_approval(request)
        choice = self._ask_choice()
        if choice == "y":
            return ApprovalOutcome(Decision.APPROVE)
        if choice == "a":
            self._approved_tools.add(request.tool_name)
            return ApprovalOutcome(Decision.APPROVE_FOR_SESSION)
        feedback = self._ask_feedback()
        return ApprovalOutcome(Decision.REJECT, feedback=feedback or None)

    def _ask_choice(self) -> str:
        while True:
            try:
                answer = self._prompt.prompt(_PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Treat an aborted prompt as a rejection, the safe default.
                return "n"
            if answer in {"y", "yes"}:
                return "y"
            if answer in {"a", "all", "always"}:
                return "a"
            if answer in {"n", "no", ""}:
                return "n"
            self._renderer.warning("Please answer with y, a or n.")

    def _ask_feedback(self) -> str:
        try:
            return self._prompt.prompt(
                "Optional: tell the assistant what to do instead: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return ""
