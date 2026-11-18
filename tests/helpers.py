"""Assertions on the tool result envelope shared by the test modules."""

from typing import Any


def assert_ok(result: dict) -> Any:
    """Assert a success envelope and return its ``data``."""
    assert set(result) == {"success", "data", "error"}
    assert result["success"] is True
    assert result["error"] is None
    assert result["data"] is not None
    return result["data"]


def assert_err(result: dict, code: str) -> str:
    """Assert an error envelope with the given code and return its message."""
    assert set(result) == {"success", "data", "error"}
    assert result["success"] is False
    assert result["data"] is None
    error = result["error"]
    assert set(error) == {"code", "message"}
    assert error["code"] == code
    assert type(error["code"]) is str  # plain str, not the StrEnum, so it serialises cleanly
    assert isinstance(error["message"], str) and error["message"]
    assert "Traceback" not in error["message"]
    return error["message"]
