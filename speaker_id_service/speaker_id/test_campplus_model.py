import unittest
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from campplus_model import get_nonlinear

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

if __name__ == '__main__':
    unittest.main()
