import sys
import os
import pytest

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speaker_id_service.speaker_id.campplus_model import get_nonlinear

def test_get_nonlinear_invalid_config():
    """Test that get_nonlinear raises ValueError for unexpected modules."""
    with pytest.raises(ValueError) as exc_info:
        # Pass an unexpected module name 'invalid_module'
        get_nonlinear('invalid_module', channels=64)

    assert "Unexpected module (invalid_module)" in str(exc_info.value)
