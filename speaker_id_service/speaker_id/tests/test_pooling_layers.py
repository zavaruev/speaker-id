import pytest
import torch
import sys
import os

# Add the parent directory to the Python path to import pooling_layers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pooling_layers import ASTP

def test_astp_3d_input():
    in_dim = 128
    batch_size = 4
    time_steps = 100

    astp = ASTP(in_dim=in_dim)

    # ASTP expects (B, F, T) which is (batch_size, in_dim, time_steps)
    x = torch.randn(batch_size, in_dim, time_steps)

    out = astp(x)

    # Output should be (B, 2*F) -> mean and std combined
    expected_shape = (batch_size, 2 * in_dim)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

    # Assert no NaNs are produced
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_astp_global_context_att():
    in_dim = 128
    batch_size = 4
    time_steps = 100

    astp = ASTP(in_dim=in_dim, global_context_att=True)

    x = torch.randn(batch_size, in_dim, time_steps)
    out = astp(x)

    expected_shape = (batch_size, 2 * in_dim)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_astp_get_out_dim():
    in_dim = 128
    astp = ASTP(in_dim=in_dim)

    out_dim = astp.get_out_dim()
    assert out_dim == 2 * in_dim, f"Expected out_dim {2*in_dim}, got {out_dim}"
