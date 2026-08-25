import sys
import os
import torch
import pytest
import unittest
import torch.nn as nn

# Add the parent directory to the Python path to import campplus_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campplus_model import CAMPPlus, get_nonlinear, DenseLayer

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

class TestDenseLayer(unittest.TestCase):

    def test_initialization(self):
        layer = DenseLayer(in_channels=128, out_channels=256, bias=True)
        self.assertIsInstance(layer, DenseLayer)
        self.assertEqual(layer.linear.in_channels, 128)
        self.assertEqual(layer.linear.out_channels, 256)
        self.assertIsNotNone(layer.linear.bias)

    def test_forward_3d_input(self):
        layer = DenseLayer(in_channels=128, out_channels=256)
        layer.eval()
        x = torch.randn(32, 128, 100)
        out = layer(x)
        self.assertEqual(out.shape, (32, 256, 100))

    def test_forward_2d_input(self):
        layer = DenseLayer(in_channels=128, out_channels=256)
        layer.eval()
        x = torch.randn(32, 128)
        out = layer(x)
        self.assertEqual(out.shape, (32, 256))


class TestGetNonlinear(unittest.TestCase):

    def test_relu(self):
        channels = 64
        model = get_nonlinear('relu', channels)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 1)
        self.assertIn('relu', model._modules)
        self.assertIsInstance(model.relu, nn.ReLU)
        self.assertTrue(model.relu.inplace)

    def test_prelu(self):
        channels = 64
        model = get_nonlinear('prelu', channels)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 1)
        self.assertIn('prelu', model._modules)
        self.assertIsInstance(model.prelu, nn.PReLU)
        self.assertEqual(model.prelu.num_parameters, channels)

    def test_batchnorm(self):
        channels = 64
        model = get_nonlinear('batchnorm', channels)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 1)
        self.assertIn('batchnorm', model._modules)
        self.assertIsInstance(model.batchnorm, nn.BatchNorm1d)
        self.assertEqual(model.batchnorm.num_features, channels)
        self.assertTrue(model.batchnorm.affine)

    def test_batchnorm_no_affine(self):
        channels = 64
        model = get_nonlinear('batchnorm_', channels)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 1)
        self.assertIn('batchnorm', model._modules)
        self.assertIsInstance(model.batchnorm, nn.BatchNorm1d)
        self.assertEqual(model.batchnorm.num_features, channels)
        self.assertFalse(model.batchnorm.affine)

    def test_composed_string(self):
        channels = 64
        model = get_nonlinear('batchnorm-relu', channels)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 2)

        modules = list(model._modules.items())

        self.assertEqual(modules[0][0], 'batchnorm')
        self.assertIsInstance(modules[0][1], nn.BatchNorm1d)
        self.assertEqual(modules[0][1].num_features, channels)

        self.assertEqual(modules[1][0], 'relu')
        self.assertIsInstance(modules[1][1], nn.ReLU)
        self.assertTrue(modules[1][1].inplace)

    def test_invalid_string(self):
        channels = 64
        with self.assertRaises(ValueError) as context:
            get_nonlinear('invalid', channels)
        self.assertIn("Unexpected module (invalid)", str(context.exception))

class TestTDNNLayer(unittest.TestCase):

    def test_instantiation(self):
        in_channels = 16
        out_channels = 32
        kernel_size = 3

        from campplus_model import TDNNLayer
        layer = TDNNLayer(in_channels, out_channels, kernel_size)

        self.assertIsInstance(layer, nn.Module)
        self.assertIsInstance(layer.linear, nn.Conv1d)
        self.assertEqual(layer.linear.in_channels, in_channels)
        self.assertEqual(layer.linear.out_channels, out_channels)
        self.assertEqual(layer.linear.kernel_size[0], kernel_size)

        self.assertIsInstance(layer.nonlinear, nn.Sequential)

    def test_forward_pass(self):
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = 2
        seq_len = 50

        from campplus_model import TDNNLayer
        import torch
        layer = TDNNLayer(in_channels, out_channels, kernel_size)

        x = torch.randn(batch_size, in_channels, seq_len)
        output = layer(x)

        # Expected sequence length after Conv1d with kernel_size=3, stride=1, padding=0
        expected_seq_len = seq_len - kernel_size + 1

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, out_channels, expected_seq_len))


class TestFCM(unittest.TestCase):

    def test_initialization(self):
        from campplus_model import FCM, BasicResBlock
        m_channels = 32
        feat_dim = 80
        num_blocks = [2, 2]

        model = FCM(block=BasicResBlock, num_blocks=num_blocks, m_channels=m_channels, feat_dim=feat_dim)

        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(model.conv1, nn.Conv2d)
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertEqual(model.conv1.out_channels, m_channels)
        self.assertEqual(model.conv1.kernel_size, (3, 3))

        self.assertIsInstance(model.layer1, nn.Sequential)
        self.assertIsInstance(model.layer2, nn.Sequential)
        self.assertEqual(len(model.layer1), num_blocks[0])
        self.assertEqual(len(model.layer2), num_blocks[1])

        self.assertIsInstance(model.conv2, nn.Conv2d)
        self.assertEqual(model.conv2.in_channels, m_channels)
        self.assertEqual(model.conv2.out_channels, m_channels)
        self.assertEqual(model.conv2.stride, (2, 1))

        expected_out_channels = m_channels * (feat_dim // 8)
        self.assertEqual(model.out_channels, expected_out_channels)

    def test_forward_pass(self):
        from campplus_model import FCM, BasicResBlock
        import torch

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

        self.assertIsInstance(output, torch.Tensor)
        expected_out_channels = m_channels * (feat_dim // 8)
        self.assertEqual(output.shape, (batch_size, expected_out_channels, seq_len))


if __name__ == '__main__':
    unittest.main()

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

def test_get_nonlinear_invalid_config():
    """Test that get_nonlinear raises ValueError for unexpected modules."""
    with pytest.raises(ValueError) as exc_info:
        # Pass an unexpected module name 'invalid_module'
        get_nonlinear('invalid_module', channels=64)

    assert "Unexpected module (invalid_module)" in str(exc_info.value)
