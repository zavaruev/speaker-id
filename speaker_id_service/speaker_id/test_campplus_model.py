import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from campplus_model import get_nonlinear, DenseLayer

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
