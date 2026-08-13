import torch
import pytest

from pooling_layers import TSTP, MQMHASTP

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
