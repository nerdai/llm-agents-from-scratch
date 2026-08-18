"""Utility functions for tools."""

from typing import Any

from jsonschema import SchemaError, ValidationError, validate

from ..data_structures import ToolCall, ToolCallErrorDetails


def validate_tool_call_arguments(
    tool_call: ToolCall,
    schema: dict[str, Any],
) -> ToolCallErrorDetails | None:
    """Validate a tool call's arguments against a JSON schema.

    Args:
        tool_call: The tool call whose arguments to validate.
        schema: The JSON schema to validate against.

    Returns:
        ``None`` if the arguments are valid, otherwise
        ``ToolCallErrorDetails`` describing the validation failure.
    """
    try:
        validate(tool_call.arguments, schema=schema)
    except (SchemaError, ValidationError) as e:
        return ToolCallErrorDetails(
            error_type=e.__class__.__name__,
            message=e.message,
        )
    return None
