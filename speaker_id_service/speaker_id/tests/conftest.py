import os
import pytest
from unittest.mock import patch

# Set default API key for tests so that get_api_key does not fail
os.environ["API_KEY"] = "default_secret_key"

@pytest.fixture(autouse=True)
def mock_os_path_getsize():
    with patch('os.path.getsize', return_value=100) as mock:
        yield mock
