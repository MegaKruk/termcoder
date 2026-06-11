# Designing a Privacy-First, Terminal-Native Agentic Coding Assistant — Research Findings, Manifesto & Design Document

## TL;DR
- Build a **single, well-equipped agent** (LLMs autonomously using tools in a loop) on a **custom micro-framework** that wraps **LiteLLM** for provider abstraction, rather than adopting LangGraph wholesale — but borrow LangGraph's `interrupt()`/checkpointer patterns for human-in-the-loop. Reserve sub-agents (Anthropic's orchestrator-worker, Cognition's "read-broad, write-single-threaded") only for read-only sub-tasks like codebase exploration.
- For codebase understanding, lead with **agentic search (ripgrep + tree-sitter repo map à la Aider)** — Anthropic removed RAG/vector search from Claude Code in May 2025 because agentic search "outperformed everything, by a lot" — and offer an **optional** local embedding layer (LanceDB + nomic-embed-code, AST/cAST chunking) for very large repos.
- For safety, run the agent on the host with strict workspace path-validation, and execute all agent-generated code in **rootless Podman** (daemonless, rootless-by-default, no `docker.sock` escalation path); gate every non-read-only action behind a Copilot/Claude-Code-style diff approval. Manage long sessions with **auto-compaction at ~80-92% of the context window**.
- Web search comes nearly free: **LiteLLM also exposes a unified `search()` API** across Tavily, Exa, Brave, Perplexity, Serper, Bing, Google PSE, and self-hostable **SearXNG**. Ship it as the flagship reference *skill* in **Phase 4 (post-MVP)**, defaulting to privacy-preserving SearXNG, paired with **trafilatura** (optionally Crawl4AI) for reading fetched pages. Treat all fetched web content as untrusted (a prompt-injection surface).

---

## PROJECT MANIFESto

**Vision.** A fully open-source, terminal-native agentic coding assistant — a "Copilot in your shell" — that a developer can trust to work inside a single project workspace, that can run entirely on locally-hosted models, and whose every consequential action is reviewed before it happens. It should be as extensible as Goose (MCP + skills), as careful as Claude Code (permission gate + compaction), and as inspectable as Aider (git-native, plain-text state).

**Core principles.**
1. **Privacy by default.** Every dependency is open source. The tool runs end-to-end against local models (Ollama / llama.cpp) with no data leaving the machine. Cloud providers are opt-in. Memory and history are local, plain-text, and inspectable.
2. **Safety as architecture, not decoration.** The agent has no sudo, is confined to a designated workspace by path validation, and runs all generated code in a rootless container that cannot escalate to host root. Every file write, delete, or command execution requires explicit user accept/reject. Errors are fed back to the model, never silently swallowed.
3. **Extensibility through open standards.** New model providers are a config entry (LiteLLM). New capabilities are MCP servers or Agent-Skills-style `SKILL.md` folders loaded with progressive disclosure. The agent loop is small and deterministic; capability lives at the edges.
4. **Coherence over cleverness.** A single-threaded write path keeps behavior consistent and debuggable. Complexity (sub-agents, semantic search, graph memory) is added only when evidence on real repos justifies its token and maintenance cost.

**Non-goals.** Not a fully-autonomous "fire and forget" cloud agent; not an IDE/GUI; not a hosted SaaS; not a multi-tenant platform; not a place to run untrusted third-party code without explicit sandbox upgrades; not a kitchen-sink framework. We will not build clever multi-agent writer swarms, and we will not relitigate the host-runs-agent / container-runs-code split or the rootless-Podman preference — those are settled.

---

## Key Findings

### 1. The harness matters far more than the model
A source-level analysis of Claude Code v2.1.88 (~1,900 TS files, ~512K LoC) found only **1.6% of the codebase is AI decision logic; 98.4% is deterministic infrastructure** — permission gates, context management, tool routing, recovery. The core agent loop is "a simple while-loop" (`query.ts`, ~1,729 lines). The lesson for our project: invest engineering in the harness (permissions, context, tools, sandbox), not in clever prompt orchestration.

### 2. Single-agent is the right default for coding; multi-agent is for breadth-first read tasks
- **Cognition ("Don't Build Multi-Agents," June 2025):** parallel writer-agents make conflicting implicit decisions and produce fragile output; they advocate single-threaded agents with a separate compression LLM for context. Their March 2026 follow-up ("Multi-Agents: What's Actually Working") narrowed the exception to "multiple agents contribute intelligence while **writes stay single-threaded**."
- **Anthropic's multi-agent research system:** per Anthropic's engineering blog "How we built our multi-agent research system," "a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval" (the example task: "identify all the board members of the companies in the Information Technology S&P 500"). Anthropic notes these systems "use about 15× more tokens than chats." Crucially, Anthropic reports "three factors explained 95% of the performance variance in the BrowseComp evaluation… token usage by itself explains 80% of the variance, with the number of tool calls and the model choice as the two other explanatory factors."
- **Anthropic's stated boundary on when multi-agent fails:** "domains that require all agents to share the same context or involve many dependencies between agents are not a good fit for multi-agent systems today" — which describes most coding work.
- **Synthesis (Phil Schmid):** the real axis is **read vs. write**. Parallelize reads (exploration, search); keep writes single-threaded.
- Claude Code itself uses a single-threaded master loop and spawns sub-agents (the `Task` tool) only for scoped, isolated-context sub-jobs that "return a condensed result to the main agent."

### 3. Agentic search has largely displaced RAG for coding agents
- Boris Cherny (Claude Code creator, Latent Space, May 2025): "We tried very early versions… that actually used RAG… Eventually we landed on just agentic search… it outperformed everything. By a lot." Cursor, Windsurf, Cline, Devin, Sourcegraph Amp followed.
- An Amazon Science paper (AAAI 2026, arXiv:2602.23368) measured agentic keyword search at 94.5% of RAG faithfulness with zero vector store.
- Nuance (Cursor, mid-2025): Cursor still uses vector search **alongside** grep. The synthesis (GTC 2026): give the agent an extra `search_code` tool but let it decide when to use it vs. plain grep/find.
- Aider's **repo map** is the gold standard for structural context: tree-sitter extracts definition/reference tags across 130+ languages (via `tree-sitter-language-pack`), builds a graph (files = nodes, references = edges), runs **PageRank** to rank symbols, and fits the most important signatures into a token budget (default 1,024 tokens via `--map-tokens`). Per DeepWiki (Aider-AI/aider, repomap.py487-514), edge weights use "multipliers for mentioned identifiers (10x), well-named identifiers (10x), and chat files (50x)," and the map is fitted to budget via binary search. Aider (created by Paul Gauthier, launched May 2023) processes ~15B tokens/week with this system (per the Tekai catalog, April 2026: "43k+ GitHub stars, 5.7M PyPI installs, and processes approximately 15B tokens per week"). It is ~600 lines of core logic and uses `grep-ast`, `networkx`, and SQLite mtime-based caching.

### 4. Context-window management: auto-compaction is table stakes
Claude Code's "Compressor" auto-compacts at roughly **80-92% of the context window** (sources cite ~80% / ~92% / ~95% depending on version — an arXiv eval paper pins Claude Code v2.1.50 at ~80% / ~160K of 200K, and Codex CLI at ~90% / ~245K of 272K). It summarizes prior turns into a `<summary>` block, drops earlier messages, and continues. Anthropic shipped this as a public API beta (`compact-2026-01-12`) with a configurable `input_tokens` trigger. Key risks: summarization is lossy (tool-call outputs and details said only in chat are lost); best practice is to compact proactively (~60-70%) and allow user-directed `/compact [instructions]`. CLAUDE.md-style files survive because they're re-read from disk.

### 5. MCP is the de-facto extensibility standard; Agent Skills is the new portable knowledge layer
- **MCP** (Anthropic, Nov 2024) is now adopted by OpenAI, Google, Microsoft. It's a JSON-RPC client-server protocol exposing **tools, resources, and prompts** over stdio / SSE / Streamable HTTP. The official **Python SDK** (`mcp`, with `FastMCP`) builds both clients and servers; `langchain-mcp-adapters` and `mcp-agent` (lastmile) provide higher-level integration. Streamable HTTP is the recommended production transport.
- **Security caveat ("the S in MCP stands for Security"):** tool poisoning, cross-server tool shadowing, prompt injection. Tools are "model-controlled by design" but the spec requires human-in-the-loop. Tools like MCP-scan and the "naive API-to-MCP conversion (Hold)" caution from Thoughtworks apply.
- **Agent Skills** (Anthropic open standard, Dec 18 2025; adopted by OpenAI, Google, GitHub Copilot, Cursor): a `SKILL.md` folder with YAML frontmatter (`name`, `description`) + markdown body + optional `scripts/`, `references/`, `assets/`. The design principle is **progressive disclosure**: only name+description (~30-80 tokens each) load at startup; the full body loads on match; bundled files/scripts load only during execution. Guidance: keep `SKILL.md` under ~500 lines / ~5k tokens. This is an ideal model for our skill/plugin system.

### 6. Sandboxing: rootless Podman is the best fit for our threat model
- **Both Docker and Podman share the host kernel** — neither alone is sufficient against a kernel exploit (NIST SP 800-190). For true isolation you need microVMs (Firecracker, libkrun) or gVisor.
- **The Docker daemon socket (`/var/run/docker.sock`) is "one of the most common container escape vectors in the wild"** — membership in the `docker` group is effectively root. **Podman has no daemon and is rootless by default**, mapping container UID 0 to an unprivileged host user via user namespaces. A rootless container escape lands as your unprivileged user, not root. Podman also drops to 11 default capabilities (vs Docker's 14) and integrates SELinux/AppArmor/seccomp.
- This directly serves the project's stated goal of preventing the agent from using the Docker daemon as a privilege-escalation path. **Recommendation: target rootless Podman as the primary backend, with rootless Docker as a fallback; both via the `docker`/`podman` Python SDK or subprocess.**
- Hardening recipe (confirmed from Podman/seccomp docs): `--network none`, `--read-only` + small `--tmpfs /tmp`, `--memory`, `--cpus`, `--pids-limit`, `--cap-drop ALL`, `--security-opt no-new-privileges`, custom seccomp profile, non-root user inside, mount workspace `:ro` or scoped `:rw`. Capture `stdout`/`stderr`/exit code via `container.wait()` + `container.logs()` (demux streams), or use a library like **epicbox** (one-time containers, `cputime`/`memory` limits, returns `{exit_code, stdout, stderr, timeout, oom_killed}`).
- OpenHands' Runtime is a strong reference: a client-server **Docker** runtime that builds an "OH runtime image," launches a container, runs an action-execution server inside, and returns observations. OpenHands V1 (arXiv:2511.03690) moved to "**optional isolation** — the agent runs locally by default but can switch to a sandboxed environment," and "stateless by default, one source of truth for state."
- Codex CLI is the deepest local sandbox: written in **Rust** (`codex-rs`, Apache 2.0), it uses **Apple Seatbelt (`sandbox-exec`) on macOS** and **Bubblewrap + seccomp (with Landlock as supplementary/fallback) on Linux**; sandbox tiers are `read-only` / `workspace-write` / `danger-full-access`, with approval presets `--suggest` / `--auto-edit` / `--full-auto`; network off by default; writes confined to the workspace via bwrap `--bind` writable roots. OpenAI's docs: "Codex CLI is OpenAI's coding agent that you can run locally from your terminal… It's open source and built in Rust for speed and efficiency."

### 7. Provider abstraction: LiteLLM is the obvious foundation
**LiteLLM** gives one `completion()` call across 100+ providers (OpenAI, Anthropic, Gemini, Mistral, plus local **Ollama** and llama.cpp), normalizing responses to the OpenAI ChatCompletions format, with streaming, tool/function calling, retries, fallbacks, and cost tracking. For Ollama it supports `ollama_chat/` prefix with `supports_function_calling: true`. It's already the default LLM layer for CrewAI, OpenHands, and others. This solves privacy/local-model support and provider-agnostic tool-calling in one dependency.

### 8. Orchestration frameworks: real tradeoffs
- **LangGraph** — graph/state-machine; first-class **checkpointing**, **durable execution**, and **`interrupt()`** for human-in-the-loop (pause, persist to checkpointer keyed by `thread_id`, resume with `Command(resume=...)`; note the node **re-executes from the top** on resume, so side effects must go *after* `interrupt()`). Powerful but heavy, LangChain-coupled, with a learning curve. The developer's prior positive experience is a real asset.
- **PydanticAI** — type-safe, MIT, model-agnostic (20+ providers), v1.0 (Sept 2025) API-stability commitment, native MCP. Most concise (~160 LoC for a sample chat app vs LangGraph's ~280). Best DX for Python teams; less ergonomic for very dynamic multi-agent flows.
- **OpenAI Agents SDK** — minimal `Agent`/`Runner`, handoffs, guardrails, sessions, MCP (incl. hosted MCP, `require_approval`). Provider-agnostic but OpenAI-centric.
- **CrewAI** — role-based multi-agent; fast prototyping; heavier (~420 LoC sample).
- **AutoGen/AG2** — conversational multi-agent; the Microsoft v0.4 rewrite vs community AG2 v0.2 split is a maintenance risk.
- **Smolagents** — minimal, code-centric; **mcp-agent** — MCP-native, programmatic control flow ("write `if`/`while`, not graphs"), Temporal durable execution.
- Anthropic's own guidance ("Building Effective Agents"): "find the simplest solution possible… many frameworks… make it easy to get started but… add layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug." They recommend starting with direct LLM API calls. Their distilled definition: "agents are LLMs autonomously using tools in a loop."

### 9. Memory: separate per-chat history from project memory; start file-based
- The field separates **per-chat conversation history** (the message log, compacted as it grows) from **long-term project memory** (conventions, architecture decisions, preferences, learned facts).
- Claude Code's persistent anchor is `CLAUDE.md` — a markdown file re-read every session — plus auto-memory. GitHub Copilot has "Copilot Memory" storing repo conventions with citations.
- Dedicated tools: **Mem0** (library; abstracts the LLM provider, works with Ollama; per Mem0's research page (paper arXiv:2504.19413, ECAI-accepted): a "26% relative uplift in overall LLM-as-a-Judge score over OpenAI's memory feature—66.9% versus 52.9%," "slashes p95 latency by 91% (1.44s vs. 17.12s)," and a "90% reduction in token consumption, requiring only ~1.8K tokens per conversation compared to 26K for full-context methods"), **Letta/MemGPT** (stateful runtime treating context like OS memory paging; pluggable backends), **Zep** (temporal knowledge graph via Graphiti; async summarization), **Cognee** (graph). For a privacy-first, local-first tool, the strongest first move is **file-based, inspectable markdown memory** (à la Claude Code) plus optional SQLite FTS; add a vector/Mem0 backend later if needed.

### 10. Terminal UI: Textual + Rich, or prompt_toolkit + Rich
- **Rich** = styled output (tables, syntax highlighting, markdown, diffs). **Textual** = full TUI framework on top of Rich (widgets, CSS, reactive state, async event loop, `App.run_test()` Pilot testing). **prompt_toolkit** = best for line-editing REPLs (powers IPython, pgcli; Aider uses it). **Typer** = CLI arg parsing.
- Aider's stack: prompt_toolkit input, streamed styled markdown output, LiteLLM, git auto-commit. For a chat/REPL that also renders rich diffs, prompt_toolkit (input) + Rich (output) is the pragmatic baseline; Textual is the choice for a more ambitious full-screen app.

---

## Details

### Architecture of GitHub Copilot CLI (GA Feb 2026)
Copilot CLI (`npm install -g @github/copilot`) is "an autonomous coding agent that can plan complex tasks, execute multistep workflows, edit files, run tests, and iterate." Key mechanics directly relevant to our design:
- **Four agent modes** cycled with Shift+Tab: **Interactive** (default; every tool call needs approval), **Plan** (build a reviewed implementation plan before coding), **Autopilot** (autonomous, no per-action approval), and **Fleet** (parallel subagents).
- **Approval UX:** every file change and command execution requires explicit approval. The three-way prompt: (1) **Yes** (this once), (2) **Yes, and approve TOOL for the session**, (3) **No, and tell Copilot what to do differently (Esc)**. `--allow-all-tools`/`--yolo` bypasses. It warns that approving a whole tool (e.g., `rm`) lets it run any such command.
- **Diff/undo:** `/diff` shows syntax-highlighted inline diffs with line-level comments; Esc-Esc rewinds file changes to any session snapshot; `/undo` reverts the last turn.
- **Auto-compaction** at ~95% of the context window, in the background.
- **Repository memory** (cross-session conventions) + **MCP** (native GitHub MCP server) + **Agent Skills** + **custom agents** (`.agent.md`) + **hooks** (preToolUse can deny/modify, postToolUse post-processing).
- **Sandboxing:** local sandbox via `/sandbox enable` (filesystem/network/capability restriction) plus cloud sandboxes (public preview). Built-in specialized agents: Explore (codebase analysis), Task (builds/tests), Code Review, Plan.

### Comparable CLI/terminal agents (architecture snapshots)
- **Aider** — terminal pair-programmer; tree-sitter PageRank repo map; multiple **edit formats** (whole-file, unified diff, etc., chosen by model capability) parsed and applied by `Coder` subclasses; git auto-commit; LiteLLM; linter feedback loop. No built-in sandbox (relies on host).
- **Claude Code** — single-threaded master loop; flat append-only message history as working memory; permission gate; `Task` sub-agents with isolated context; compaction; CLAUDE.md; hooks/skills/MCP/subagents extension layers stratified by context cost.
- **OpenHands (formerly OpenDevin)** — platform with universal agent controller; Docker client-server runtime; event-sourced state; V1 SDK (arXiv:2511.03690) is modular, opt-in sandbox, model-agnostic via LiteLLM (100+ providers), with a security analyzer; 10 composable "condenser" strategies for context management; reduced system-attributable failures 61% vs V0.
- **Goose (Block; now Linux Foundation)** — local agent, CLI + desktop; **MCP is the extension system** ("all six extension types speak it"; adding an OpenAI-compatible provider is a ~10-line JSON file); deliberately minimal agent loop; relies on local system security (no built-in sandbox) but has a 5-inspector safety pipeline.
- **Cline** — VS Code extension + CLI/SDK; human-in-the-loop approval for every file change and command; BYOK across many providers + local; MCP; has a "YOLO mode."
- **Codex CLI** — Rust, OS-native sandbox (Seatbelt / bwrap+seccomp+Landlock); summarize-and-replace compaction with encrypted summaries; approval presets.
- **Continue.dev / Cursor / Tabby / Plandex / SWE-agent** — Continue = IDE-integrated, BYOK; Cursor = RAG-like local filesystem index + grep; Tabby = self-hosted completion (Apache-2.0); Plandex = accumulates multi-file changes in a sandbox for review before applying, supports branching plans; SWE-agent = research agent with a simple agent-computer interface.
- "What I learned reading 15 agent codebases" lesson: **the gap between projects that invest in context management and those that don't is the single biggest predictor of whether an agent survives a 2-hour session.** And: "Either you commit to multi-layer security or you effectively have none. Minimum viable: a sandbox for code execution, human approval for writes, and loop detection."

### Codebase understanding & retrieval (recommended subsystem)
1. **Structural repo map (primary):** tree-sitter (`tree-sitter` + `tree-sitter-language-pack`, pre-built wheels) extracting `name.definition.*` / `name.reference.*` tags → `networkx` PageRank → token-budgeted signature map. Cache by mtime in SQLite. Inject when working in a git repo.
2. **Agentic search (primary):** give the agent `ripgrep`-backed `search`, `find`, `read_file`, and `ls` tools and let it iterate. This is what Claude Code/Cursor/Cline converged on.
3. **Optional semantic layer (large repos):** **AST/cAST chunking** (the cAST method, arXiv:2506.15655, EMNLP 2025 Findings — a recursive split-then-merge over tree-sitter AST nodes that "respects syntactic integrity, packs each chunk to a fixed size budget, is language-invariant, and reproduces the original file verbatim when concatenated"; reference implementations: `astchunk` and Supermemory's `code-chunk`) → embeddings via **nomic-embed-code** (7B, GPU; weights/data/eval fully open) or **nomic-embed-text v1.5** (CPU, "the best model you can run everywhere") / **Qwen3-Embedding** via Ollama → **LanceDB** (embedded, serverless, Rust, Apache-2.0 — "an open-source embedded vector database… runs in-process with zero-copy access to data… without a running server," the best fit for a local-first embedded Python app; ChromaDB Apache-2.0 is a close runner-up, Qdrant is more server-oriented). Expose as one optional `semantic_search` tool the agent may call.

### Conversation + compaction subsystem
- Per-chat JSONL transcript on disk (Claude Code writes to `~/.claude/projects/*.jsonl`, enabling resume/fork/rewind). Each project/workspace can hold many chats; chats are isolated from project memory.
- Token counting via `tiktoken` (estimate) or provider-reported usage (accurate). Trigger compaction at a configurable threshold (~70-80% to stay ahead of degradation). Summarize earlier turns into a `<summary>` block preserving decisions, file states, and task context; keep recent turns verbatim. Allow `/compact [instructions]`. Snapshot affected files before edits to enable rewind.

### Sandboxed execution subsystem
- Agent process runs on host; only agent-generated code runs in a container. **Rootless Podman primary** (`podman` Python bindings or subprocess), rootless Docker fallback. Per-run ephemeral container; workspace bind-mounted; `--network none` by default (toggle for dependency installs); `--cap-drop ALL`, `--memory`, `--cpus`, `--pids-limit`, `no-new-privileges`, custom seccomp. Capture stdout/stderr/exit code; enforce wall-clock timeout; detect OOM kill. Document that for fully untrusted code, microVM (libkrun via Podman, Firecracker, E2B/Daytona) is the stronger tier.

### Workspace confinement (host filesystem)
- All file tools validate paths: resolve to absolute real path (`os.path.realpath`), assert it is within the configured workspace root, reject symlink traversal and `..` escapes, deny absolute paths outside root. No sudo; agent runs as the unprivileged invoking user. This is the host-side guardrail complementing the container.

### File-edit diff & approval subsystem
- Represent edits as unified diffs / hunks (mirror Aider's edit formats and Copilot's `/diff`). Render with Rich syntax-highlighted diffs. Present accept/reject (and accept-all-for-session) like Copilot's three-way prompt. On accept, apply atomically and snapshot for undo; on reject, feed the rejection back to the model as a tool result (errors-are-feedback pattern). Map this cleanly onto LangGraph-style `interrupt()` if LangGraph is used, or a custom pause/resume if not.

### Module / package structure (proposed)
```
agent/
  core/        # the while-loop: context assembly, model call, tool dispatch, recovery
  providers/   # LiteLLM wrapper + provider registry (openai, anthropic, gemini, ollama)
  tools/       # read/list/grep/find/edit/run + tool schema (Pydantic)
  permissions/ # the gate: classify read-only vs mutating; approval prompts
  context/     # token counting, compaction, system-prompt assembly
  repomap/     # tree-sitter + networkx PageRank + SQLite cache
  search/      # ripgrep wrapper; optional LanceDB semantic layer
  memory/      # project markdown + SQLite FTS (pluggable Mem0/graph later)
  sandbox/     # rootless Podman runner (epicbox-style ephemeral containers)
  sessions/    # per-chat JSONL store, resume/fork/rewind, snapshots
  skills/      # SKILL.md loader (progressive disclosure)
  mcp/         # MCP client (official `mcp` SDK)
  tui/         # prompt_toolkit input + Rich rendering (diffs, streaming)
  subagents/   # read-only Explorer (isolated context, condensed return)
  cli.py       # Typer entrypoint
```

---

## Recommendations

**Orchestration — build a custom micro-framework, borrow LangGraph patterns.** Given (a) the finding that 98.4% of a production agent is deterministic harness, (b) Anthropic's explicit advice to avoid heavy abstractions, (c) the need for total control over the agent loop, approval interrupts, compaction, and provider-agnostic streaming, and (d) the privacy goal of minimal open-source dependencies — a thin custom loop (`while not done: assemble_context → call_model(LiteLLM, stream) → parse_tool_calls → permission_gate → execute → append_result`) is superior to adopting LangGraph wholesale. **However**, prototype the human-in-the-loop layer with LangGraph's `interrupt()` + SQLite checkpointer first (the developer knows it), and only port to custom code if dependency weight or control becomes a problem. Decision threshold: if you find yourself fighting LangGraph's state model for streaming or approval UX, switch to custom; if it accelerates you, keep it.

**Single agent, with read-only sub-agents.** Ship a single agent for all writes. Add an isolated-context **Explorer sub-agent** (read-only: grep/read/repo-map) that returns a condensed summary — this is the one multi-agent pattern with evidence behind it for coding, and it keeps the main context lean. Defer a web-search sub-agent and a conversational/coder split until benchmarks justify the ~15× token cost.

**Concrete library stack:**
- LLM/provider: **LiteLLM** (cloud + Ollama/llama.cpp local).
- Structured tools/validation: **Pydantic** (+ optionally **instructor** for structured outputs).
- Repo map: **tree-sitter**, **tree-sitter-language-pack**, **grep-ast**, **networkx**.
- Search: **ripgrep** (subprocess).
- Optional semantic: **LanceDB** + AST/cAST chunking (`astchunk`) + Ollama embeddings (nomic-embed-code / nomic-embed-text / Qwen3-Embedding).
- Memory: file-based markdown (`PROJECT.md`/`CONVENTIONS.md`) + SQLite FTS; Mem0 optional later.
- Sandbox: **rootless Podman** (primary) via Python SDK; epicbox-style ephemeral runner.
- TUI: **prompt_toolkit** (input) + **Rich** (output/diffs); Textual if going full-screen.
- MCP: official **`mcp`** Python SDK (FastMCP) for client + skill servers.
- Tokens: **tiktoken**.
- CLI: **Typer**.

**Phased roadmap:**
1. **MVP (host-only):** REPL (prompt_toolkit+Rich) → single agent loop over LiteLLM (one cloud + Ollama) → core tools (read/list/grep/edit) → workspace path validation → unified-diff approval → per-chat JSONL history. No container yet (edits approved; code run only on explicit approval in host with warning).
2. **Safety & context:** rootless Podman execution subsystem; token counting + auto-compaction; file snapshots/undo.
3. **Understanding:** tree-sitter PageRank repo map; ripgrep agentic search tools; project markdown memory.
4. **Extensibility:** MCP client integration; Agent-Skills-style `SKILL.md` plugin loader with progressive disclosure; provider registry for new models.
5. **Scale/optional:** read-only Explorer sub-agent; optional LanceDB semantic search; optional Mem0/graph memory; web-search tool; microVM tier for untrusted code.

**Benchmarks that would change the plan:** if internal evals show the Explorer sub-agent doesn't beat single-agent grep on your repos, drop it. If compaction quality degrades tasks, lower the trigger threshold and add user-directed compaction. If LanceDB semantic search doesn't beat agentic grep on retrieval accuracy for your corpus sizes, keep it off by default.

---

## Caveats
- **Source recency/reliability:** This is a fast-moving space (mid-2026). Several architectural details of Claude Code come from community reverse-engineering of a leaked npm source map (v2.1.88) and from secondary blogs, not official docs — treat specifics like the "h2A dual-buffer queue," "nO master loop," and exact compaction percentages (80% vs 92% vs 95%) as approximate and version-dependent. Official Anthropic docs confirm the loop, permissions, compaction, and sub-agent concepts in general terms.
- **Compaction thresholds conflict across sources** (80%/83.5%/92%/95%); they're version- and product-dependent. Make ours configurable.
- **Containers are not a security boundary against kernel exploits** — rootless Podman dramatically reduces the privilege-escalation blast radius but does not fully isolate untrusted code; document the microVM upgrade path.
- **MCP introduces a real supply-chain attack surface** (tool poisoning, shadowing, prompt injection); only load skills/MCP servers from trusted sources and audit them.
- **Multi-agent token cost** (~15×) and coordination fragility are real; the single-agent default is deliberately conservative.
- **Codex CLI's Linux sandbox is now primarily Bubblewrap + seccomp with Landlock as a supplementary/fallback layer** (the architecture shifted from the earlier "Landlock + seccomp" framing); both are present.
- Some cited comparison blogs (vector DB roundups, framework rankings, "Project Polaris" Copilot news) carry vendor or speculative framing; the core facts (licenses, embeddability, MCP/skills standards, sandbox mechanisms) are corroborated across multiple independent and primary sources.