import torch
import pytest
from pooling_layers import TAP

def test_tap_forward():
    # Test TAP forward pass
    # Input shape: (batch_size, in_dim, time_steps)
    # E.g. (2, 4, 3)
    x = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
    ])

    # Expected: mean across the last dimension (time_steps)
    expected_mean = x.mean(dim=-1).flatten(start_dim=1)

    tap = TAP(in_dim=4)
    output = tap(x)

    assert torch.allclose(output, expected_mean)
    assert output.shape == (2, 4)

def test_tap_get_out_dim():
    tap = TAP(in_dim=128)
    assert tap.get_out_dim() == 128

    tap = TAP(in_dim=512)
    assert tap.get_out_dim() == 512
