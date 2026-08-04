"""Subagents constants."""

CATALOG_SPEC_TEMPLATE = """
  <subagent>
    <name>{name}</name>
    <description>{description}</description>
  </subagent>
""".strip()

# default recipe names + step caps (subagents/recipes.py)
GENERAL_NAME = "general"
EXPLORE_NAME = "explore"
GENERAL_MAX_STEPS = 20
EXPLORE_MAX_STEPS = 10
