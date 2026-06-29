import sys
import os
import torch
import pytest

# Add the parent directory to the Python path to import campplus_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campplus_model import CAMPPlus

def test_campplus_initialization():
    """Test that the CAMPPlus model initializes successfully with default/common parameters."""
    model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func='TSTP')

    assert model is not None
    assert model.head.out_channels == 32 * 10 # 32 * (80 // 8)
    assert model.pool_out_dim == 1024 # 512 * 2 for TSTP
    assert model.xvector.dense.linear.out_channels == 512

def test_campplus_forward_pass():
    """Test the forward pass of the CAMPPlus model with a dummy tensor."""
    model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func='TSTP')
    model.eval()

    # Create a dummy tensor of shape (batch_size, time_steps, feat_dim)
    # The __main__ block in campplus_model.py uses (1, 200, 80)
    batch_size = 1
    time_steps = 200
    feat_dim = 80
    dummy_input = torch.zeros(batch_size, time_steps, feat_dim)

    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify the output shape
    # The expected output shape is (batch_size, embed_dim)
    expected_embed_dim = 512
    assert output.shape == (batch_size, expected_embed_dim)

def test_campplus_get_frame_level_feat():
    """Test the get_frame_level_feat method of the CAMPPlus model."""
    model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func='TSTP')
    model.eval()

    batch_size = 1
    time_steps = 200
    feat_dim = 80
    dummy_input = torch.zeros(batch_size, time_steps, feat_dim)

    with torch.no_grad():
        output = model.get_frame_level_feat(dummy_input)

    # Check that output is a 3D tensor: (batch_size, time_steps // 2, channels)
    # The TDNN layer has stride=2, which halves the time dimension
    assert len(output.shape) == 3
    assert output.shape[0] == batch_size
    assert output.shape[1] == time_steps // 2
