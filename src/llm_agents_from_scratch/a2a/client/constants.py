"""A2A constants."""

CATALOG_SPEC_TEMPLATE = """
  <a2a_agent>
    <name>{name}</name>
    <description>{description}</description>{skills}
  </a2a_agent>
""".strip()

CATALOG_INDIVIDUAL_A2A_SKILL_TEMPLATE = "      <a2a_skill>{name}</a2a_skill>"

CATALOG_A2A_SKILLS_TEMPLATE = """
    <a2a_skills>
{skills}
    </a2a_skills>"""

A2A_INPUT_REQUIRED_TEMPLATE = """The A2A agent '{name}' needs more \
information before it can continue:

{question}

To continue, call `from_scratch__use_a2a_agent` again with name='{name}', \
task_id='{task_id}', and `task` set to the requested information. This \
resumes the same remote task rather than starting a new one.""".strip()
