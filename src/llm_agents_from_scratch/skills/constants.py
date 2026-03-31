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
The user wants to use a specific skill to perform this task.

Use the from_scratch__use_skill tool with `name={name}` to activate the
skill which loads the information on the skill as well as potentially any
resources associated with the skill.

<additional-user-prompt>
{prompt}
</additional-user-prompt>
""".strip()
