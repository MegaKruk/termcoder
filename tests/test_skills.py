"""Tests for SKILL.md loading and progressive disclosure.

These verify the three tiers in isolation: discovery and the startup catalog
(tier 1), the read_skill tool loading a body on demand (tier 2), and that
frontmatter parsing tolerates the simple cases the standard allows while
rejecting skills that lack a name or description.
"""

from __future__ import annotations

from termcoder.approval.auto import AutoApprover
from termcoder.skills import ReadSkillTool, SkillRegistry, discover_skills, load_skill
from termcoder.tools.base import ToolContext
from termcoder.workspace.paths import WorkspaceGuard


def _make_skill(root, name, description, body="Body text."):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return folder


def _context(tmp_path):
    return ToolContext(workspace=WorkspaceGuard(tmp_path), approver=AutoApprover())


def test_load_skill_parses_frontmatter(tmp_path):
    folder = _make_skill(tmp_path, "format-code", "Format code with the project tools.")
    skill = load_skill(folder / "SKILL.md")

    assert skill is not None
    assert skill.name == "format-code"
    assert skill.description == "Format code with the project tools."
    assert "Body text." in skill.body
    assert skill.root == folder


def test_load_skill_requires_name_and_description(tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: only-name\n---\n\nBody.", encoding="utf-8")
    assert load_skill(folder / "SKILL.md") is None


def test_load_skill_handles_quoted_values(tmp_path):
    folder = tmp_path / "quoted"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        '---\nname: "quoted-name"\ndescription: \'A quoted description.\'\n---\n\nBody.',
        encoding="utf-8",
    )
    skill = load_skill(folder / "SKILL.md")
    assert skill is not None
    assert skill.name == "quoted-name"
    assert skill.description == "A quoted description."


def test_discover_skills_finds_subfolders_and_sorts(tmp_path):
    _make_skill(tmp_path, "zebra", "Last alphabetically.")
    _make_skill(tmp_path, "alpha", "First alphabetically.")
    skills = discover_skills([tmp_path])
    assert [skill.name for skill in skills] == ["alpha", "zebra"]


def test_discover_skills_first_directory_wins(tmp_path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _make_skill(primary, "shared", "Primary version.")
    _make_skill(secondary, "shared", "Secondary version.")

    skills = discover_skills([primary, secondary])

    assert len(skills) == 1
    assert skills[0].description == "Primary version."


def test_registry_catalog_lists_names_and_descriptions(tmp_path):
    _make_skill(tmp_path, "build", "Build the project.")
    _make_skill(tmp_path, "test", "Run the tests.")
    registry = SkillRegistry.from_directories([tmp_path])

    catalog = registry.catalog()

    assert catalog is not None
    assert "build: Build the project." in catalog
    assert "test: Run the tests." in catalog
    assert "read_skill" in catalog


def test_empty_registry_has_no_catalog():
    registry = SkillRegistry()
    assert registry.catalog() is None
    assert len(registry) == 0


def test_read_skill_tool_returns_body(tmp_path):
    _make_skill(tmp_path, "deploy", "Deploy the service.", body="Step one. Step two.")
    registry = SkillRegistry.from_directories([tmp_path])
    tool = ReadSkillTool(registry)

    result = tool.execute(tool.args_model(name="deploy"), _context(tmp_path))

    assert result.ok
    assert "Step one. Step two." in result.content
    assert "Skill: deploy" in result.content


def test_read_skill_tool_reports_unknown_skill(tmp_path):
    registry = SkillRegistry.from_directories([tmp_path])
    tool = ReadSkillTool(registry)

    result = tool.execute(tool.args_model(name="missing"), _context(tmp_path))

    assert not result.ok
    assert "No skill named 'missing'" in result.content
