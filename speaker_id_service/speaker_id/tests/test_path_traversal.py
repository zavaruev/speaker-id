import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Create a dummy model directory to avoid urllib downloads during module import
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Mock the entire urllib and torch to prevent any downloads or heavy initializations during import
with patch("urllib.request.urlretrieve", return_value=None), \
     patch("torch.load", return_value={}), \
     patch("torch.cuda.is_available", return_value=False), \
     patch("campplus_model.CAMPPlus.load_state_dict", MagicMock()):
    from app import app

from fastapi.testclient import TestClient

client = TestClient(app)

class TestSpeakerID(unittest.TestCase):
    def test_identify_path_traversal(self):
        # We simulate a file upload with a path traversal payload.
        # It should not attempt to write outside of /tmp directory with the exact requested filename.
        with patch("builtins.open", MagicMock()) as mock_open:
            with patch("shutil.copyfileobj", MagicMock()):
                with patch("app.convert_to_wav", return_value=False):
                    with patch("torch.load", return_value={}):
                        response = client.post(
                            "/identify",
                            files={"file": ("../../../etc/passwd", b"dummy audio content", "audio/wav")}
                        )

        # Check what the filename passed to `open` was.
        # It should just be "passwd" because of os.path.basename.
        mock_open.assert_called()
        args, kwargs = mock_open.call_args
        opened_path = args[0]

        # Verify that os.path.basename was applied: "passwd" should be in the path, but not "../"
        self.assertNotIn("../", opened_path)
        self.assertTrue(opened_path.endswith("_passwd"))

    def test_enroll_path_traversal(self):
        with patch("builtins.open", MagicMock()) as mock_open:
            with patch("shutil.copyfileobj", MagicMock()):
                with patch("app.convert_to_wav", return_value=False):
                    with patch("torch.load", return_value={}):
                        response = client.post(
                            "/enroll",
                            data={"user_id": "test_user"},
                            files={"files": ("../../../var/log/syslog", b"dummy audio content", "audio/wav")}
                        )

        mock_open.assert_called()
        args, kwargs = mock_open.call_args
        opened_path = args[0]
        self.assertNotIn("../", opened_path)
        self.assertTrue(opened_path.endswith("_syslog"))
