# termcoder

A privacy-first, terminal-native agentic coding assistant. It is an interactive
chat REPL with a single agent loop, a small set of file and shell tools, strict
workspace confinement, diff-based approval for every change, per-chat history
stored as plain text, a sandbox for shell commands, automatic conversation
compaction, and undo for file edits.

termcoder is built to run fully locally against an Ollama model so no code or
prompt leaves your machine. Cloud models are optional and only activate when
you set the matching API key.

## What is included

Core assistant:

- Interactive REPL built on prompt_toolkit and Rich.
- A single agent loop over LiteLLM that works with local Ollama models and
  cloud models (OpenAI, Anthropic) through one code path.
- Core tools: read_file, list_directory, search_text (ripgrep with a built-in
  fallback), write_file, edit_file, and run_command.
- Workspace confinement: every path is validated and cannot escape the
  workspace directory.
- Approval gate: every file write and shell command shows a diff or the exact
  command and must be approved. You can approve once, approve a tool for the
  whole session, or reject with feedback.
- Per-chat sessions saved as JSON Lines, with resume support.

Safety and context:

- Sandboxed shell execution. By default run_command runs inside an ephemeral,
  rootless Podman container (Docker is a fallback). The container drops all
  capabilities, disables new privileges, has no network unless you opt in, and
  is limited in memory, CPU and process count. The workspace is mounted so
  edits and build output are visible on the host. If no container engine is
  available, commands run on the host and the approval prompt says so.
- Token counting and automatic compaction. As a session approaches the model's
  context window, older turns are summarized and recent turns are kept verbatim.
  The full transcript on disk is never altered; the summary is folded into the
  system prompt at request time. Compact on demand with /compact.
- File snapshots and undo. The file changes from each turn are captured, so
  /undo reverts the most recent turn: prior contents are restored and files the
  agent newly created are removed. Note that run_command effects are not
  snapshotted, since a command can change anything.

Token economy:

- Usage metering. Every model call is counted, including compaction summaries.
  After each turn a dim line shows tokens in and out (plus cache hits), and
  /usage reports session totals with an estimated cost for cloud models. Set
  show_usage = false in the config to hide the per-turn line.
- Prompt-cache friendly by construction. The system prompt is static and the
  conversation is append-only between compactions, so provider prompt caches
  hit naturally. For Anthropic models termcoder adds cache_control breakpoints
  automatically (cached input is billed at roughly a tenth of the normal
  price); OpenAI caches stable prefixes on its own at roughly half price. Per
  model this can be turned off with cache_prompts = false.
- Cheap summaries. Compaction can use a different, cheaper model via
  context.summary_model, since the summary request is the largest single
  prompt termcoder ever sends.
- Retry on malformed output. Small local models occasionally emit broken
  tool-call JSON; termcoder retries the request up to two times instead of
  failing the whole turn.

## Requirements

- Python 3.14 or newer.
- One model backend:
  - Local (default, recommended for privacy): Ollama running with a
    tool-capable model. For example:
    ```
    ollama pull llama3.1
    ```
    The model must support tool or function calling. llama3.1 does.
  - Cloud (optional): set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
- Optional: ripgrep (`rg`) on your PATH for faster search. Without it, a
  built-in Python scanner is used.
- Optional but recommended: a rootless container engine for the command
  sandbox. Rootless Podman is the default; Docker also works. Pull the sandbox
  image once (for example `podman pull python:3.14-slim`) or enable network for
  the engine so it can pull on first use. Without an engine, commands run on the
  host.
- Optional: tiktoken for a closer token estimate used by compaction. Without it,
  a character-based estimate is used. Install with `pip install -e ".[tokenizers]"`.

## Install

From the project root:

```
pip install -e .
```

This installs the `termcoder` command.

## Run

Start a chat in the current directory (this becomes the workspace):

```
termcoder
```

Or point it at a specific workspace and pick a model:

```
termcoder chat --workspace /path/to/project --model anthropic
```

List configured models or past sessions:

```
termcoder models
termcoder sessions
```

You can also run it without installing:

```
python -m termcoder
```

### In-chat commands

```
/help            Show help.
/new             Start a new chat session.
/sessions        List chat sessions for this workspace.
/resume <id>     Resume a previous session by id.
/model [name]    Show or switch the active model.
/compact [focus] Summarize older turns now to free context space.
/usage           Show token and cost usage for this session.
/undo            Revert the file changes from the most recent turn.
/tools           List the available tools.
/clear           Clear the screen.
/exit, /quit     Leave termcoder.
```

## Configuration

Settings are optional and live in `.termcoder/config.toml` inside the
workspace. Anything you do not set falls back to a sensible default, and the
default model is local Ollama. See `.termcoder/config.example.toml` for the
available keys, including how to define extra models, choose the sandbox
backend and image, allow command network access, tune compaction thresholds,
pick a cheaper summary model, harden the container with a read-only root
filesystem, control prompt caching and the usage readout, and turn the
run_command tool or undo off entirely.

API keys are never stored in the config file. The config only names the
environment variable that holds each key.

## Test

```
pip install -e ".[dev]"
pytest
```

The test suite runs offline. It does not require any model, API key, or network
access, and does not import the model provider layer.

## E2E Test
To run end-to-end test (example):

```
export OPENAI_API_KEY=sk-proj-...
termcoder chat -w /home/megakruk/workspace/python/termcoder-sandbox -m gpt
```

## Project layout

```
src/termcoder/
  config.py        Configuration and the model registry.
  errors.py        Shared exception types.
  workspace/       Path validation that confines all file access.
  approval/        Approval types and unified diff generation (no UI code).
  tools/           The tool framework and the built-in tools.
  sandbox/         Command runners: host and rootless container, with a factory.
  context/         Token counting and conversation compaction.
  snapshots/       File snapshots and undo.
  llm/             Chat message helpers.
  providers/       LiteLLM client and setup (the only place LiteLLM is used).
  sessions/        Per-chat JSON Lines storage.
  agent/           The agent loop and the system prompt.
  ui/              Rich rendering, the approval prompt, and the REPL.
  cli.py           The command-line entry point.
```

This structure leaves clear seams for later phases. The sandbox exposes a single
runner protocol, so a stronger isolation tier (such as a microVM) can be added
without touching the tool. A repository map and web search slot in as new
modules, and MCP and sub-agents as additional tool sources.
