"""Errors for Agent Skills."""

from .core import LLMAgentsFromScratchError, LLMAgentsFromScratchWarning


class SkillsWarning(LLMAgentsFromScratchWarning):
    """Base warning for all skill-related warnings."""

    pass


class SkillValidationWarning(SkillsWarning):
    """Emitted when a skill directory has validation issues."""

    pass


class SkillsError(LLMAgentsFromScratchError):
    """Base error for all skill-related exceptions."""

    pass


class SkillValidationError(SkillsError):
    """Raised when validating a skill produces an error."""

    pass


class MissingSkillMdError(SkillValidationError):
    """Raised when a skill directory does not contain a SKILL.md file."""

    pass


class InvalidFrontmatterError(SkillValidationError):
    """Raised when a SKILL.md file has invalid or unparseable frontmatter."""

    pass


class NameMismatchError(SkillValidationError):
    """Raised when the skill name does not match its parent directory name."""

    pass
