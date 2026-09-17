import os
import io
import tempfile
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers
from benchmark import (
    create_large_file,
    simulate_concurrent_requests_blocking,
    simulate_concurrent_requests_nonblocking,
    measure_event_loop_lag,
)

def test_create_large_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dummy.txt")
        create_large_file(path, size_mb=1)
        assert os.path.exists(path)
        assert os.stat(path).st_size == 1 * 1024 * 1024

@pytest.fixture
def dummy_upload_files():
    files = []
    for _ in range(3):
        f = io.BytesIO(b"dummy audio data")
        upload_file = UploadFile(filename="test.wav", file=f, headers=Headers())
        files.append(upload_file)
    return files

@pytest.mark.asyncio
async def test_simulate_concurrent_requests_blocking(dummy_upload_files):
    duration = await simulate_concurrent_requests_blocking(dummy_upload_files)
    assert duration >= 0.0

@pytest.mark.asyncio
async def test_simulate_concurrent_requests_nonblocking(dummy_upload_files):
    duration = await simulate_concurrent_requests_nonblocking(dummy_upload_files)
    assert duration >= 0.0

@pytest.mark.asyncio
async def test_measure_event_loop_lag(dummy_upload_files):
    duration, max_delay = await measure_event_loop_lag(
        simulate_concurrent_requests_nonblocking, dummy_upload_files
    )
    assert duration >= 0.0
    assert max_delay >= 0.0

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from speaker_id_service.speaker_id.pooling_layers import MHASTP as MHASTP_New

class MHASTP_Old(torch.nn.Module):
    def __init__(self, in_dim, layer_num=2, head_num=2, d_s=1, bottleneck_dim=64, **kwargs):
        super(MHASTP_Old, self).__init__()
        assert (in_dim % head_num) == 0
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for j in range(layer_num - 1):
                att_trans.add_module('att_' + str(j), nn.Conv1d(channel_dims[j], channel_dims[j + 1], 1, 1))
                att_trans.add_module('tanh' + str(j), nn.Tanh())
            att_trans.add_module('att_' + str(layer_num - 1), nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num], 1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        chunks_out = []
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-7))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

def test_mhastp_performance():
    B, C, T = 2, 512, 1000
    in_tensor = torch.randn(B, C, T)

    model_old = MHASTP_Old(C, head_num=32)
    model_new = MHASTP_New(C, head_num=32)

    # Warmup
    for _ in range(5):
        _ = model_old(in_tensor)
        _ = model_new(in_tensor)

    start_time = time.time()
    for _ in range(50):
        _ = model_old(in_tensor)
    old_time = time.time() - start_time

    start_time = time.time()
    for _ in range(50):
        _ = model_new(in_tensor)
    new_time = time.time() - start_time

    # Check that new implementation is faster (with some buffer for CI variance)
    # The expected speedup is ~1.46x
    assert new_time < old_time * 1.1
