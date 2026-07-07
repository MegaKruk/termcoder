"""Interactive approval prompt, answerable from the terminal or a remote client.

Implements the three-way choice used by GitHub Copilot: approve once, approve
this tool for the rest of the session, or reject with optional feedback. Tools
approved for the session are remembered here so the user is not asked again for
the same tool.

Without a session bus the prompt is purely local, exactly as in earlier
versions. With a bus, each request is also published to remote clients and the
first responder wins: a remote answer dismisses the terminal prompt, a terminal
answer clears the remote card, and every client sees how it was resolved.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession

from ..approval.types import ApprovalOutcome, ApprovalRequest, Decision
from ..remote.bus import EventBus
from ..remote.events import ApprovalRequestedEvent, ApprovalResolvedEvent
from .interruptible import CancelToken, PromptInterrupted, prompt_interruptible
from .renderer import Renderer

_PROMPT = "Approve? [y] once  [a] allow this tool for the session  [n] reject: "


class ConsoleApprover:
    """Ask the user to approve actions, remembering session-wide approvals."""

    def __init__(
        self,
        renderer: Renderer,
        prompt_session: PromptSession | None = None,
        bus: EventBus | None = None,
    ):
        self._renderer = renderer
        self._prompt = prompt_session or PromptSession()
        self._bus = bus
        self._approved_tools: set[str] = set()

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        if request.tool_name in self._approved_tools:
            return ApprovalOutcome(Decision.APPROVE_FOR_SESSION)

        self._renderer.render_approval(request)
        if self._bus is None:
            return self._decide_local(request)
        return self._decide_arbitrated(request, self._bus)

    # Local-only flow (no remote clients possible)

    def _decide_local(self, request: ApprovalRequest) -> ApprovalOutcome:
        choice = self._ask_choice()
        if choice == "y":
            return ApprovalOutcome(Decision.APPROVE)
        if choice == "a":
            self._approved_tools.add(request.tool_name)
            return ApprovalOutcome(Decision.APPROVE_FOR_SESSION)
        feedback = self._ask_feedback()
        return ApprovalOutcome(Decision.REJECT, feedback=feedback or None)

    # Arbitrated flow (terminal and remote race, first responder wins)

    def _decide_arbitrated(
        self, request: ApprovalRequest, bus: EventBus
    ) -> ApprovalOutcome:
        request_id = bus.open_approval()
        bus.publish(
            ApprovalRequestedEvent(
                request_id=request_id,
                tool_name=request.tool_name,
                summary=request.summary,
                detail=request.detail,
                detail_kind=request.detail_kind,
                destructive=request.destructive,
                note=request.note,
            )
        )
        cancel = CancelToken()
        bus.watch_approval(request_id, cancel.trip)

        choice = self._ask_choice(cancel)
        if choice is not None:
            bus.resolve_approval(
                request_id, self._outcome_for(choice), resolved_by="terminal"
            )
        outcome, resolved_by = bus.wait_for_decision(request_id)

        if resolved_by == "terminal" and outcome.decision is Decision.REJECT:
            feedback = self._ask_feedback()
            outcome = ApprovalOutcome(Decision.REJECT, feedback=feedback or None)
        if outcome.decision is Decision.APPROVE_FOR_SESSION:
            self._approved_tools.add(request.tool_name)

        bus.publish(
            ApprovalResolvedEvent(
                request_id=request_id,
                decision=outcome.decision.value,
                resolved_by=resolved_by,
            )
        )
        if resolved_by != "terminal":
            self._renderer.info(f"Resolved from {resolved_by}: {outcome.decision.value}")
        return outcome

    @staticmethod
    def _outcome_for(choice: str) -> ApprovalOutcome:
        """Map a terminal keypress to an approval outcome, without feedback."""
        if choice == "y":
            return ApprovalOutcome(Decision.APPROVE)
        if choice == "a":
            return ApprovalOutcome(Decision.APPROVE_FOR_SESSION)
        return ApprovalOutcome(Decision.REJECT)

    def _ask_choice(self, cancel: CancelToken | None = None) -> str | None:
        """Prompt for y, a or n; None means a remote client answered first."""
        while True:
            try:
                answer = prompt_interruptible(
                    self._prompt, _PROMPT, cancel
                ).strip().lower()
            except PromptInterrupted:
                return None
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
