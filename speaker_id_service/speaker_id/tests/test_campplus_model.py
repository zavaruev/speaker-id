import sys
import os
import torch
import torch.nn as nn
import pytest

# Add the parent directory to the Python path to import campplus_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campplus_model import CAMPPlus, get_nonlinear

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

def test_get_nonlinear_relu():
    channels = 32
    nonlinear = get_nonlinear('relu', channels)

    assert isinstance(nonlinear, nn.Sequential)
    assert len(nonlinear) == 1
    assert isinstance(nonlinear[0], nn.ReLU)
    assert nonlinear[0].inplace is True
    assert list(nonlinear._modules.keys())[0] == 'relu'

def test_get_nonlinear_prelu():
    channels = 64
    nonlinear = get_nonlinear('prelu', channels)

    assert isinstance(nonlinear, nn.Sequential)
    assert len(nonlinear) == 1
    assert isinstance(nonlinear[0], nn.PReLU)
    assert nonlinear[0].num_parameters == channels
    assert list(nonlinear._modules.keys())[0] == 'prelu'

def test_get_nonlinear_batchnorm():
    channels = 128
    nonlinear = get_nonlinear('batchnorm', channels)

    assert isinstance(nonlinear, nn.Sequential)
    assert len(nonlinear) == 1
    assert isinstance(nonlinear[0], nn.BatchNorm1d)
    assert nonlinear[0].num_features == channels
    assert nonlinear[0].affine is True
    assert list(nonlinear._modules.keys())[0] == 'batchnorm'

def test_get_nonlinear_batchnorm_no_affine():
    channels = 256
    nonlinear = get_nonlinear('batchnorm_', channels)

    assert isinstance(nonlinear, nn.Sequential)
    assert len(nonlinear) == 1
    assert isinstance(nonlinear[0], nn.BatchNorm1d)
    assert nonlinear[0].num_features == channels
    assert nonlinear[0].affine is False
    assert list(nonlinear._modules.keys())[0] == 'batchnorm'

def test_get_nonlinear_chained():
    channels = 64
    nonlinear = get_nonlinear('batchnorm-relu', channels)

    assert isinstance(nonlinear, nn.Sequential)
    assert len(nonlinear) == 2

    assert isinstance(nonlinear[0], nn.BatchNorm1d)
    assert nonlinear[0].num_features == channels
    assert nonlinear[0].affine is True
    assert list(nonlinear._modules.keys())[0] == 'batchnorm'

    assert isinstance(nonlinear[1], nn.ReLU)
    assert nonlinear[1].inplace is True
    assert list(nonlinear._modules.keys())[1] == 'relu'

def test_get_nonlinear_invalid():
    channels = 32
    with pytest.raises(ValueError, match=r'Unexpected module \(invalid\)\.'):
        get_nonlinear('invalid', channels)

def test_get_nonlinear_chained_with_invalid():
    channels = 32
    with pytest.raises(ValueError, match=r'Unexpected module \(unknown\)\.'):
        get_nonlinear('relu-unknown', channels)
