"""Utility functions for tools."""

import json
from typing import Any

from jsonschema import SchemaError, ValidationError, validate

from ..data_structures import ToolCall, ToolCallResult


def validate_tool_call_arguments(
    tool_call: ToolCall,
    schema: dict[str, Any],
) -> ToolCallResult | None:
    """Validate a tool call's arguments against a JSON schema.

    Args:
        tool_call: The tool call whose arguments to validate.
        schema: The JSON schema to validate against.

    Returns:
        ``None`` if the arguments are valid, otherwise an error
        ``ToolCallResult`` describing the validation failure.
    """
    try:
        validate(tool_call.arguments, schema=schema)
    except (SchemaError, ValidationError) as e:
        error_details = {
            "error_type": e.__class__.__name__,
            "message": e.message,
        }
        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=json.dumps(error_details),
            error=True,
        )
    return None
