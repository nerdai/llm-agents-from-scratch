"""A2A constants."""

CATALOG_SPEC_TEMPLATE = """
  <a2a_agent>
    <name>{name}</name>
    <description>{description}</description>{skills}
  </a2a_agent>
""".strip()

CATALOG_A2A_SKILL_TEMPLATE = "      <a2a_skill>{name}</a2a_skill>"
