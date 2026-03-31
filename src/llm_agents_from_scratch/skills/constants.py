"""Skills constants."""

MAX_NAME_LENGTH = 64

CATALOG_SKILL_TEMPLATE = """
  <skill>
    <name>{name}</name>
    <description>{description}</description>
    <location>{location}</location>
  </skill>
""".strip()

OPTIONAL_SUBDIRS = ["assets", "scripts", "references"]

SKILL_RESOURCES_TEMPLATE = """
Relative paths in this skill are relative to the skill directory.
<skill_resources>
{files}
</skill_resources>
""".strip()

ACTIVATION_CONTENT_TEMPLATE = """
<skill_content name="{name}">
{body}

Skill directory: {skill_dir}
{skill_resources}</skill_content>""".strip()

EXPLICIT_SKILL_ACTIVATION_TEMPLATE = """
Activate the '{name}' skill by calling the from_scratch__use_skill tool,
then follow its instructions.
""".strip()

EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE = """
Activate the '{name}' skill by calling the from_scratch__use_skill tool,
then follow its instructions to complete the task below.

The user has provided the following instructions to guide your work:
<user-instructions>{prompt}</user-instructions>
""".strip()
