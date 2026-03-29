import re


def clean_text(text: str | None) -> str:
    """Normalize text for model input.

    - Strips leading/trailing whitespace
    - Collapses multiple whitespace into single spaces
    - Removes non-printable characters
    """
    if not text:
        return ""
    text = re.sub(r"[^\S ]+", " ", text)  # non-space whitespace → space
    text = re.sub(r" {2,}", " ", text)     # collapse multiple spaces
    return text.strip()


def build_task_prompt(title: str, description: str | None, tags: list[str]) -> str:
    """Build a structured prompt string from task fields.

    Used by the decomposition model to understand the task context.
    """
    parts = [f"Task: {clean_text(title)}"]
    desc = clean_text(description)
    if desc:
        parts.append(f"Description: {desc}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(parts)


def build_member_profile(
    name: str,
    skills: list[str],
    skill_levels: dict[str, str],
    role: str,
) -> str:
    """Build a text profile for a team member (used by assigner embedding)."""
    skill_parts = []
    for s in skills:
        level = skill_levels.get(s, "intermediate")
        skill_parts.append(f"{s} ({level})")
    skills_str = ", ".join(skill_parts) if skill_parts else "no listed skills"
    return f"{name} | role: {role} | skills: {skills_str}"
