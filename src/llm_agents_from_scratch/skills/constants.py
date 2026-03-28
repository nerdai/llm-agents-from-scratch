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

SKILL_RESOURCES_TEMPLATE = """\
Relative paths in this skill are relative to the skill directory.
<skill_resources>
{files}
</skill_resources>
"""

ACTIVATION_CONTENT_TEMPLATE = """\
<skill_content name="{name}">
{body}

Skill directory: {skill_dir}
{skill_resources}</skill_content>"""
