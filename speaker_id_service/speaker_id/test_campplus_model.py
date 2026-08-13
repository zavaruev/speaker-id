import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from campplus_model import get_nonlinear, DenseLayer, CAMLayer

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

class TestCAMLayer(unittest.TestCase):

    def test_initialization(self):
        bn_channels = 16
        out_channels = 32
        kernel_size = 3
        layer = CAMLayer(bn_channels=bn_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=1, padding=1,
                         dilation=1, bias=False)
        self.assertIsInstance(layer, CAMLayer)
        self.assertEqual(layer.linear_local.in_channels, bn_channels)
        self.assertEqual(layer.linear_local.out_channels, out_channels)
        self.assertEqual(layer.linear_local.kernel_size[0], kernel_size)

    def test_forward_3d_input(self):
        bn_channels = 16
        out_channels = 32
        layer = CAMLayer(bn_channels=bn_channels, out_channels=out_channels,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, bias=False)
        layer.eval()

        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, bn_channels, time_steps)
        out = layer(x)
        self.assertEqual(out.shape, (batch_size, out_channels, time_steps))

    def test_forward_2d_input(self):
        bn_channels = 16
        out_channels = 32
        layer = CAMLayer(bn_channels=bn_channels, out_channels=out_channels,
                         kernel_size=3, stride=1, padding=1,
                         dilation=1, bias=False)
        layer.eval()

        time_steps = 50
        x = torch.randn(bn_channels, time_steps)
        out = layer(x)
        self.assertEqual(out.shape, (out_channels, time_steps))

    def test_seg_pooling(self):
        layer = CAMLayer(bn_channels=16, out_channels=32, kernel_size=3,
                         stride=1, padding=1, dilation=1, bias=False)

        x = torch.randn(2, 16, 50)

        # Test avg pooling
        out_avg = layer.seg_pooling(x, stype='avg')
        self.assertEqual(out_avg.shape, (2, 16, 50))

        # Test max pooling
        out_max = layer.seg_pooling(x, stype='max')
        self.assertEqual(out_max.shape, (2, 16, 50))

        # Test invalid pooling type
        with self.assertRaises(ValueError):
            layer.seg_pooling(x, stype='invalid')


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

if __name__ == '__main__':
    unittest.main()
