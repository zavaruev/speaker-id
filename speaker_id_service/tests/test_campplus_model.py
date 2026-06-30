import pytest
import torch.nn as nn
import sys
import os

# Add speaker_id directory to path to allow importing campplus_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../speaker_id')))
from campplus_model import get_nonlinear

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
