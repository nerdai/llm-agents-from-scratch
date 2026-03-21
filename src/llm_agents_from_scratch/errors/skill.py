"""Errors for Agent Skills."""

from .core import LLMAgentsFromScratchError, LLMAgentsFromScratchWarning


class SkillsWarning(LLMAgentsFromScratchWarning):
    """Base warning for all skill-related warnings."""

    pass


class SkillValidationWarning(SkillsWarning):
    """Emitted when a skill has cosmetic issues but is still loaded."""

    pass


class SkillSkippedWarning(SkillsWarning):
    """Emitted when a skill is skipped due to a fatal validation error."""

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


class NameMismatchWarning(SkillValidationWarning):
    """Emitted when the skill name does not match its parent directory name."""

    pass
