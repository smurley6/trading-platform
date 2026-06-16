"""ApiKeyAuth dependency: open vs enforced, header parsing."""

import pytest
from fastapi import HTTPException

from trading.interface.auth import ApiKeyAuth, _extract_key


def test_extract_key_prefers_bearer() -> None:
    assert _extract_key("Bearer abc", "xyz") == "abc"
    assert _extract_key("bearer abc", None) == "abc"  # case-insensitive scheme


def test_extract_key_falls_back_to_x_api_key() -> None:
    assert _extract_key(None, "xyz") == "xyz"
    assert _extract_key("Basic zzz", "xyz") == "xyz"  # non-bearer auth ignored


def test_extract_key_none_when_absent_or_empty() -> None:
    assert _extract_key(None, None) is None
    assert _extract_key("Bearer ", "  ") is None


async def test_open_mode_permits_any_request() -> None:
    auth = ApiKeyAuth(None)
    assert auth.enabled is False
    # No exception regardless of headers.
    await auth(authorization=None, x_api_key=None)


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
async def test_blank_api_key_is_treated_as_open(blank: str) -> None:
    # `API_KEY=` in .env resolves to an empty string, not None — it must mean "open", never
    # "enforce a key nothing can present" (which would 401 every write).
    auth = ApiKeyAuth(blank)
    assert auth.enabled is False
    await auth(authorization=None, x_api_key=None)


async def test_enforced_mode_accepts_valid_key() -> None:
    auth = ApiKeyAuth("secret")
    assert auth.enabled is True
    await auth(authorization="Bearer secret", x_api_key=None)
    await auth(authorization=None, x_api_key="secret")


async def test_enforced_mode_rejects_invalid_or_missing() -> None:
    auth = ApiKeyAuth("secret")
    with pytest.raises(HTTPException) as missing:
        await auth(authorization=None, x_api_key=None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        await auth(authorization="Bearer nope", x_api_key=None)
    assert wrong.value.status_code == 401
