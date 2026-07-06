"""Global pytest configuration — disables the slowapi rate limiter for all tests."""

import pytest
from api.app import app


@pytest.fixture(autouse=True)
def disable_rate_limiter() -> None:
    """Globally disables the slowapi rate limiter for all pytest runs.

    Prevents 429 Too Many Requests errors during test suites where
    endpoints are hit faster than the configured rate limits allow.
    Re-enables the limiter after each test as a safety measure.
    """
    app.state.limiter.enabled = False
    yield
    app.state.limiter.enabled = True
