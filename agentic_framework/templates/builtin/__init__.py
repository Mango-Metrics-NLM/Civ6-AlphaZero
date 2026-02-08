"""
Built-in Templates for Agentic Coding System

Contains pre-configured templates for common use cases.
"""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent

# Template file paths
AGENTIC_PROMPT_YAML = TEMPLATE_DIR / "agentic_prompt_template.yaml"
AGENTIC_PROMPT_MD = TEMPLATE_DIR / "agentic_prompt_template.md"


def get_template_path(template_name: str) -> Path:
    """Get the path to a built-in template.

    Args:
        template_name: Name of the template (without extension).

    Returns:
        Path to the template file.

    Raises:
        FileNotFoundError: If template doesn't exist.
    """
    # Try YAML first, then Markdown
    yaml_path = TEMPLATE_DIR / f"{template_name}.yaml"
    if yaml_path.exists():
        return yaml_path

    md_path = TEMPLATE_DIR / f"{template_name}.md"
    if md_path.exists():
        return md_path

    raise FileNotFoundError(f"Template not found: {template_name}")


def list_templates() -> list[str]:
    """List all available built-in templates.

    Returns:
        List of template names.
    """
    templates = set()

    for ext in [".yaml", ".yml", ".md"]:
        for path in TEMPLATE_DIR.glob(f"*{ext}"):
            if not path.name.startswith("_"):
                templates.add(path.stem)

    return sorted(templates)


__all__ = [
    "AGENTIC_PROMPT_MD",
    "AGENTIC_PROMPT_YAML",
    "TEMPLATE_DIR",
    "get_template_path",
    "list_templates",
]
