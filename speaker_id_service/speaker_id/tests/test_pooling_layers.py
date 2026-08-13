import torch
import pytest

from pooling_layers import TSTP, ASP

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
