import torch
import pytest

from pooling_layers import TSTP, MHASTP, MQMHASTP, ASP, XI, TAP, TSDP, ASTP

def test_asp_init_w2v():
    """Test ASP initialization with W2V-style arguments."""
    model = ASP(input_dim=128, hidden_dim=64)
    assert model.feature_dim == 128
    assert model.out_dim == 256

def test_asp_init_wespeaker():
    """Test ASP initialization with WeSpeaker-style arguments."""
    model = ASP(in_planes=32, acoustic_dim=80)
    assert model.feature_dim == 2560
    assert model.out_dim == 5120

def test_asp_init_error():
    """Test ASP initialization without required arguments."""
    with pytest.raises(ValueError, match="Specify either"):
        ASP()

def test_asp_forward_shape_3d_w2v():
    """Test ASP forward shape with 3D input [B, T, D]."""
    input_dim = 128
    model = ASP(input_dim=input_dim)

    batch_size = 2
    time_steps = 50
    dummy_input = torch.randn(batch_size, time_steps, input_dim)

    output = model(dummy_input)
    assert output.shape == (batch_size, input_dim * 2)

def test_asp_forward_shape_3d_wespeaker():
    """Test ASP forward shape with 3D input [B, D, T]."""
    input_dim = 128
    model = ASP(input_dim=input_dim)

    batch_size = 2
    time_steps = 50
    dummy_input = torch.randn(batch_size, input_dim, time_steps)

    output = model(dummy_input)
    assert output.shape == (batch_size, input_dim * 2)

def test_asp_forward_shape_4d():
    """Test ASP forward shape with 4D input [B, C, F, T]."""
    model = ASP(in_planes=32, acoustic_dim=80)

    batch_size = 2
    channels = 32
    freq = 80
    time_steps = 50

    dummy_input = torch.randn(batch_size, channels, freq, time_steps)
    output = model(dummy_input)
    assert output.shape == (batch_size, 2560 * 2)

def test_asp_forward_values():
    """Test ASP forward mathematically with mocked attention."""
    import unittest.mock

    input_dim = 2
    model = ASP(input_dim=input_dim)

    def dummy_attention(x):
        # We need weights that sum to 1 over the time dimension (dim=2)
        # For a tensor of shape (batch_size, input_dim, time_steps)
        # where time_steps is 4, each weight should be 0.25.
        return torch.full_like(x, 0.25)

    class DummyAttention(torch.nn.Module):
        def forward(self, x):
            return dummy_attention(x)

    # We use our DummyAttention to mock the attention module properly
    model.attention = DummyAttention()
    dummy_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                 [1.0, 1.0, 1.0, 1.0]]])

    output = model(dummy_input)

    expected_mean = torch.tensor([[2.5, 1.0]])
    expected_std = torch.tensor([[1.1180339, 0.003162277]])

    expected_output = torch.cat((expected_mean, expected_std), dim=1)

    assert torch.allclose(output, expected_output, atol=1e-5)

def test_xi_forward_shape_no_stddev():
    """Test that the XI pooling layer outputs the correct shape when stddev=False."""
    in_dim = 80
    model = XI(in_dim=in_dim, stddev=False)

    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, in_dim, time_steps)

    output = model(dummy_input)

    assert output.shape == (batch_size, in_dim)

def test_xi_forward_shape_with_stddev():
    """Test that the XI pooling layer outputs the correct shape when stddev=True."""
    in_dim = 80
    model = XI(in_dim=in_dim, stddev=True)

    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, in_dim, time_steps)

    output = model(dummy_input)

    assert output.shape == (batch_size, in_dim * 2, 1)

def test_xi_invalid_input_dimensions():
    """Test that the XI pooling layer asserts on invalid input dimensions."""
    in_dim = 80
    model = XI(in_dim=in_dim)

    # 4D input
    dummy_input = torch.randn(4, 1, in_dim, 100)
    with pytest.raises(AssertionError):
        model(dummy_input)

    # 2D input
    dummy_input = torch.randn(4, in_dim)
    with pytest.raises(AssertionError):
        model(dummy_input)

def test_xi_invalid_feature_dimension():
    """Test that the XI pooling layer asserts on mismatching feature dimension."""
    in_dim = 80
    model = XI(in_dim=in_dim)

    # Valid 3D shape, but wrong feature dimension
    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, in_dim + 1, time_steps)

    with pytest.raises(AssertionError):
        model(dummy_input)

def test_tstp_forward_shape():
    """Test that the TSTP pooling layer outputs the correct shape."""
    in_dim = 80
    model = TSTP(in_dim=in_dim)

    # Input tensor shape: (batch_size, feature_dim, time_steps)
    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, in_dim, time_steps)

    output = model(dummy_input)

    # The output should have shape (batch_size, in_dim * 2)
    assert output.shape == (batch_size, in_dim * 2)

def test_tstp_forward_values():
    """Test that the TSTP pooling layer correctly computes mean and std."""
    in_dim = 2
    model = TSTP(in_dim=in_dim)

    batch_size = 1
    time_steps = 4

    # Create a specific input tensor to manually verify the values
    # Tensor shape: (1, 2, 4)
    dummy_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                 [1.0, 1.0, 1.0, 1.0]]])

    # The temporal axis is the last dimension (dim=-1)
    # Means:
    # Feature 0: (1+2+3+4)/4 = 2.5
    # Feature 1: (1+1+1+1)/4 = 1.0

    # Variances (using unbiased variance by default in PyTorch, meaning N-1 in denominator):
    # Feature 0: ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 3
    #          = (2.25 + 0.25 + 0.25 + 2.25) / 3 = 5.0 / 3 = 1.6666666
    # Feature 1: 0.0

    # Std:
    # Feature 0: sqrt(1.6666666 + 1e-7) ~ 1.290994
    # Feature 1: sqrt(0.0 + 1e-7) ~ 0.000316227

    output = model(dummy_input)

    expected_mean = torch.tensor([[2.5, 1.0]])
    expected_var = torch.tensor([[5.0 / 3.0, 0.0]])
    expected_std = torch.sqrt(expected_var + 1e-7)

    expected_output = torch.cat((expected_mean, expected_std), dim=1)

    # Check if the output matches the expected values closely
    assert torch.allclose(output, expected_output, atol=1e-5)

def test_mhastp_forward_shape_3d():
    """Test that the MHASTP pooling layer outputs the correct shape for 3D input."""
    in_dim = 80
    head_num = 4
    model = MHASTP(in_dim=in_dim, head_num=head_num)

    # Input tensor shape: (batch_size, feature_dim, time_steps)
    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, in_dim, time_steps)

    output = model(dummy_input)

    # The output should have shape (batch_size, in_dim * 2)
    assert output.shape == (batch_size, in_dim * 2)

def test_mhastp_forward_shape_4d():
    """Test that the MHASTP pooling layer outputs the correct shape for 4D input."""
    # For 4D input, it flattens the channel and feature dimensions.
    # So if input is (B, C, F, T), then in_dim should be C * F.
    channels = 8
    features = 10
    in_dim = channels * features
    head_num = 4
    model = MHASTP(in_dim=in_dim, head_num=head_num)

    # Input tensor shape: (batch_size, channels, features, time_steps)
    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, channels, features, time_steps)

    output = model(dummy_input)

    # The output should have shape (batch_size, in_dim * 2)
    assert output.shape == (batch_size, in_dim * 2)

def test_mhastp_invalid_head_num():
    """Test that MHASTP raises an AssertionError if in_dim is not divisible by head_num."""
    in_dim = 80
    head_num = 3 # 80 is not divisible by 3

    with pytest.raises(AssertionError):
        MHASTP(in_dim=in_dim, head_num=head_num)

def test_mqmhastp_forward_shape():
    """Test that the MQMHASTP pooling layer outputs the correct shape for 3D input."""
    in_dim = 80
    query_num = 2
    head_num = 8
    model = MQMHASTP(in_dim=in_dim, query_num=query_num, head_num=head_num)

    # Input tensor shape: (batch_size, feature_dim, time_steps)
    batch_size = 4
    time_steps = 100
    dummy_input = torch.randn(batch_size, in_dim, time_steps)

    output = model(dummy_input)

    # The output should have shape (batch_size, in_dim * 2 * query_num)
    assert output.shape == (batch_size, in_dim * 2 * query_num)

def test_mqmhastp_forward_shape_4d():
    """Test that the MQMHASTP pooling layer correctly handles 4D inputs."""
    in_dim = 80
    channels = 4
    features = 20 # channels * features = in_dim = 80
    query_num = 2
    head_num = 8
    model = MQMHASTP(in_dim=in_dim, query_num=query_num, head_num=head_num)

    batch_size = 4
    time_steps = 100
    # Input tensor shape: (batch_size, channels, features, time_steps)
    dummy_input = torch.randn(batch_size, channels, features, time_steps)

    output = model(dummy_input)

    # The output should still have shape (batch_size, in_dim * 2 * query_num)
    assert output.shape == (batch_size, in_dim * 2 * query_num)

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
