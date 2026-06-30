import torch
import pytest
from speaker_id_service.speaker_id.pooling_layers import TAP

def test_tap_forward_3d():
    """Test TAP (Temporal Average Pooling) with 3D input."""
    # Input shape: [Batch, Feature, Time] -> [1, 2, 3]
    x = torch.tensor([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]])

    # Expected output: mean along dim=-1, then flatten(start_dim=1)
    # Means: [ (1+2+3)/3, (4+5+6)/3 ] = [2.0, 5.0]
    expected = torch.tensor([[2.0, 5.0]])

    tap = TAP(in_dim=2)
    out = tap(x)

    assert torch.allclose(out, expected)
    assert tap.get_out_dim() == 2

def test_tap_forward_4d():
    """Test TAP with 4D input."""
    # Input shape: [Batch, Channel, Feature, Time] -> [1, 1, 2, 3]
    x = torch.tensor([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]]])

    expected = torch.tensor([[2.0, 5.0]])

    tap = TAP(in_dim=2)
    out = tap(x)

    assert torch.allclose(out, expected)
    assert tap.get_out_dim() == 2

def test_tap_gradient():
    """Test if TAP maintains gradients."""
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
    tap = TAP(in_dim=2)
    out = tap(x)

    loss = out.sum()
    loss.backward()

    # The gradient should be 1/3 for each element because it's an average over 3 elements
    expected_grad = torch.tensor([[[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]])
    assert torch.allclose(x.grad, expected_grad)

def test_tap_multiple_batches():
    """Test TAP with multiple batches."""
    x = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    ])

    expected = torch.tensor([
        [2.0, 5.0],
        [20.0, 50.0]
    ])

    tap = TAP(in_dim=2)
    out = tap(x)

    assert torch.allclose(out, expected)
    assert out.shape == (2, 2)
