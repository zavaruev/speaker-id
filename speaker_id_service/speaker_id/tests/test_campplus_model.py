import sys
import os
import torch
import torch.nn as nn
import pytest

# Add the parent directory to the Python path to import campplus_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campplus_model import CAMPPlus, get_nonlinear, DenseLayer, TDNNLayer, FCM, BasicResBlock

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
    with pytest.raises(ValueError, match=r'Unexpected module \(invalid\)'):
        get_nonlinear('invalid', channels)

def test_get_nonlinear_chained_with_invalid():
    channels = 32
    with pytest.raises(ValueError, match=r'Unexpected module \(unknown\)'):
        get_nonlinear('relu-unknown', channels)

def test_dense_layer_initialization():
    layer = DenseLayer(in_channels=128, out_channels=256, bias=True)
    assert isinstance(layer, DenseLayer)
    assert layer.linear.in_channels == 128
    assert layer.linear.out_channels == 256
    assert layer.linear.bias is not None

def test_dense_layer_forward_3d_input():
    layer = DenseLayer(in_channels=128, out_channels=256)
    layer.eval()
    x = torch.randn(32, 128, 100)
    out = layer(x)
    assert out.shape == (32, 256, 100)

def test_dense_layer_forward_2d_input():
    layer = DenseLayer(in_channels=128, out_channels=256)
    layer.eval()
    x = torch.randn(32, 128)
    out = layer(x)
    assert out.shape == (32, 256)

def test_tdnn_layer_instantiation():
    in_channels = 16
    out_channels = 32
    kernel_size = 3

    layer = TDNNLayer(in_channels, out_channels, kernel_size)

    assert isinstance(layer, nn.Module)
    assert isinstance(layer.linear, nn.Conv1d)
    assert layer.linear.in_channels == in_channels
    assert layer.linear.out_channels == out_channels
    assert layer.linear.kernel_size[0] == kernel_size
    assert isinstance(layer.nonlinear, nn.Sequential)

def test_tdnn_layer_forward_pass():
    in_channels = 16
    out_channels = 32
    kernel_size = 3
    batch_size = 2
    seq_len = 50

    layer = TDNNLayer(in_channels, out_channels, kernel_size)

    x = torch.randn(batch_size, in_channels, seq_len)
    output = layer(x)

    # Expected sequence length after Conv1d with kernel_size=3, stride=1, padding=0
    expected_seq_len = seq_len - kernel_size + 1

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, out_channels, expected_seq_len)

def test_fcm_initialization():
    m_channels = 32
    feat_dim = 80
    num_blocks = [2, 2]

    model = FCM(block=BasicResBlock, num_blocks=num_blocks, m_channels=m_channels, feat_dim=feat_dim)

    assert isinstance(model, nn.Module)
    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.in_channels == 1
    assert model.conv1.out_channels == m_channels
    assert model.conv1.kernel_size == (3, 3)

    assert isinstance(model.layer1, nn.Sequential)
    assert isinstance(model.layer2, nn.Sequential)
    assert len(model.layer1) == num_blocks[0]
    assert len(model.layer2) == num_blocks[1]

    assert isinstance(model.conv2, nn.Conv2d)
    assert model.conv2.in_channels == m_channels
    assert model.conv2.out_channels == m_channels
    assert model.conv2.stride == (2, 1)

    expected_out_channels = m_channels * (feat_dim // 8)
    assert model.out_channels == expected_out_channels

def test_fcm_forward_pass():
    m_channels = 32
    feat_dim = 80
    num_blocks = [2, 2]
    batch_size = 2
    seq_len = 200

    model = FCM(block=BasicResBlock, num_blocks=num_blocks, m_channels=m_channels, feat_dim=feat_dim)
    model.eval()

    # In CAMPPlus, the input to FCM is permuted from (B, T, F) to (B, F, T)
    x = torch.randn(batch_size, feat_dim, seq_len)
    output = model(x)

    assert isinstance(output, torch.Tensor)
    expected_out_channels = m_channels * (feat_dim // 8)
    assert output.shape == (batch_size, expected_out_channels, seq_len)
