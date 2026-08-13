import torch
import pytest
from speaker_id_service.speaker_id.pooling_layers import TAP, TSDP, ASTP

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

def test_tsdp_forward_3d():
    """Test TSDP (Temporal Standard Deviation Pooling) with 3D input."""
    # Input shape: [Batch, Feature, Time] -> [1, 2, 3]
    x = torch.tensor([[[1.0, 2.0, 3.0],
                       [4.0, 6.0, 8.0]]])

    # Standard deviation with Bessel's correction (unbiased=True by default in torch.var)
    # var of [1,2,3] is 1.0 -> sqrt(1.0 + 1e-7) approx 1.0
    # var of [4,6,8] is 4.0 -> sqrt(4.0 + 1e-7) approx 2.0
    expected = torch.tensor([[1.0, 2.0]])

    tsdp = TSDP(in_dim=2)
    out = tsdp(x)

    assert torch.allclose(out, expected, atol=1e-3)

def test_astp_forward_3d():
    """Test ASTP (Attentive Statistics Pooling) with 3D input."""
    # Input shape: [Batch, Feature, Time] -> [2, 4, 10]
    torch.manual_seed(42)
    x = torch.randn(2, 4, 10)
    astp = ASTP(in_dim=4, bottleneck_dim=8, global_context_att=False)

    # We test output dimensions because exact values depend on initialized weights
    out = astp(x)
    assert out.shape == (2, 8)  # 2 * in_dim
    assert astp.get_out_dim() == 8

def test_astp_forward_4d():
    """Test ASTP with 4D input."""
    # Input shape: [Batch, Channel, Feature, Time] -> [2, 1, 4, 10]
    torch.manual_seed(42)
    x = torch.randn(2, 1, 4, 10)
    astp = ASTP(in_dim=4, bottleneck_dim=8, global_context_att=False)

    out = astp(x)
    assert out.shape == (2, 8)
    assert astp.get_out_dim() == 8

def test_astp_forward_global_context():
    """Test ASTP with global context enabled."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 10)
    astp = ASTP(in_dim=4, bottleneck_dim=8, global_context_att=True)

    out = astp(x)
    assert out.shape == (2, 8)
    assert astp.get_out_dim() == 8

def test_astp_gradient():
    """Test if ASTP maintains gradients."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 10, requires_grad=True)
    astp = ASTP(in_dim=4, bottleneck_dim=8, global_context_att=False)

    out = astp(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 4, 10)

def test_astp_deterministic():
    """Test if ASTP outputs are deterministic for the same input."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 10)
    astp = ASTP(in_dim=4, bottleneck_dim=8, global_context_att=False)

    out1 = astp(x)
    out2 = astp(x)

    assert torch.allclose(out1, out2)

def test_tsdp_forward_4d():
    """Test TSDP with 4D input."""
    # Input shape: [Batch, Channel, Feature, Time] -> [1, 1, 2, 3]
    x = torch.tensor([[[[1.0, 2.0, 3.0],
                        [4.0, 6.0, 8.0]]]])

    expected = torch.tensor([[1.0, 2.0]])

    tsdp = TSDP(in_dim=2)
    out = tsdp(x)

    assert torch.allclose(out, expected, atol=1e-3)
