"""Tests for structured markdown memory parsing.

These cover the section parsing that turns flat memory into addressable topics,
the fuzzy section lookup, and the guarantee that a flat file with no headings
keeps working exactly as before.
"""

from __future__ import annotations

from termcoder.memory import load_project_memory, parse_sections

_STRUCTURED = (
    "# Conventions\n"
    "Use 4-space indentation.\n"
    "\n"
    "# Architecture\n"
    "The agent loop lives in agent/loop.py.\n"
    "\n"
    "## Sandbox\n"
    "Commands run in Podman.\n"
    "\n"
    "# Glossary\n"
    "RAG: retrieval augmented generation.\n"
)


def test_parse_sections_splits_by_heading():
    sections = parse_sections(_STRUCTURED)
    titles = [section.title for section in sections]
    assert titles == ["Conventions", "Architecture", "Sandbox", "Glossary"]


def test_parse_sections_captures_body():
    sections = parse_sections(_STRUCTURED)
    conventions = sections[0]
    assert "4-space indentation" in conventions.body
    assert conventions.level == 1


def test_parse_sections_subsection_level():
    sections = parse_sections(_STRUCTURED)
    sandbox = next(s for s in sections if s.title == "Sandbox")
    assert sandbox.level == 2
    assert "Podman" in sandbox.body


def test_flat_file_yields_single_section():
    sections = parse_sections("Just answer in pirate speak.")
    assert len(sections) == 1
    assert sections[0].title == "Preamble"
    assert "pirate speak" in sections[0].body


def test_preamble_before_first_heading_is_kept():
    text = "Some intro text.\n\n# Real Section\nbody"
    sections = parse_sections(text)
    assert sections[0].title == "Preamble"
    assert "intro text" in sections[0].body
    assert sections[1].title == "Real Section"


def test_load_memory_exposes_sections(tmp_path):
    (tmp_path / "TERMCODER.md").write_text(_STRUCTURED, encoding="utf-8")
    memory = load_project_memory(tmp_path, ("TERMCODER.md",))

    assert memory is not None
    assert memory.section_titles() == [
        "Conventions",
        "Architecture",
        "Sandbox",
        "Glossary",
    ]


def test_find_section_exact_and_fuzzy(tmp_path):
    (tmp_path / "TERMCODER.md").write_text(_STRUCTURED, encoding="utf-8")
    memory = load_project_memory(tmp_path, ("TERMCODER.md",))

    assert memory.find_section("Glossary").title == "Glossary"
    # Fuzzy: substring match, case-insensitive.
    assert memory.find_section("convention").title == "Conventions"
    assert memory.find_section("nonexistent") is None


def test_flat_memory_file_still_loads(tmp_path):
    (tmp_path / "TERMCODER.md").write_text(
        "Answer in pirate speak. Give a bird fact.", encoding="utf-8"
    )
    memory = load_project_memory(tmp_path, ("TERMCODER.md",))

    assert memory is not None
    assert "pirate speak" in memory.text
    assert len(memory.sections) == 1
