import os
import pytest
from unittest.mock import patch

# Set default API key for tests so that get_api_key does not fail
os.environ["API_KEY"] = "default_secret_key"

@pytest.fixture(autouse=True)
def mock_os_path_getsize():
    with patch('os.path.getsize', return_value=100) as mock:
        yield mock

@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Clear the in-memory rate-limit buckets before/after each test.

    The limiter (PR #118) is global per process and keyed by client IP, while
    the whole suite talks through TestClients sharing one IP. Without this
    reset, the first ~10 requests of the suite would consume the bucket and
    every later test would spuriously get HTTP 429.
    """
    try:
        import sys
        _app = sys.modules.get('app')
        if _app is not None and hasattr(_app, '_rate_limits'):
            _app._rate_limits.clear()
    except Exception:
        pass
    yield
    try:
        import sys
        _app = sys.modules.get('app')
        if _app is not None and hasattr(_app, '_rate_limits'):
            _app._rate_limits.clear()
    except Exception:
        pass
