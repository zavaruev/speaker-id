import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Mock the entire CAM++ model and torch loading
sys.path.insert(0, os.path.abspath('speaker_id_service/speaker_id'))
with patch('urllib.request.urlretrieve'), \
     patch('torch.load', return_value={}), \
     patch('campplus_model.CAMPPlus.load_state_dict'):
    from app import app

client = TestClient(app)

def test_identify_no_file():
    response = client.post("/identify")
    assert response.status_code == 422

def test_import():
    assert app is not None
