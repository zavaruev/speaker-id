import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_os_path_getsize():
    with patch('os.path.getsize', return_value=100) as mock:
        yield mock
