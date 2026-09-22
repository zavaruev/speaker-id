import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from benchmark_enroll2 import (
    mock_rebuild_cache,
    save_embedding,
    simulate_enroll_blocking,
    simulate_enroll_nonblocking,
    measure_event_loop_lag,
)

def test_mock_rebuild_cache():
    # It does nothing, but we should cover it
    mock_rebuild_cache()

def test_save_embedding():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch("benchmark_enroll2.SPEAKERS_DIR", tmp_path):
            embedding = np.random.rand(512)
            user_id = "test_user_123"

            save_embedding(embedding, user_id)

            expected_file = tmp_path / f"{user_id}.npy"
            assert expected_file.exists()

            loaded = np.load(expected_file)
            assert np.allclose(loaded, embedding)

@pytest.mark.asyncio
async def test_simulate_enroll_blocking():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch("benchmark_enroll2.SPEAKERS_DIR", tmp_path):
            duration = await simulate_enroll_blocking(3)
            assert duration >= 0.0

            # Should have created 3 files
            files = list(tmp_path.glob("*.npy"))
            assert len(files) == 3

@pytest.mark.asyncio
async def test_simulate_enroll_nonblocking():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch("benchmark_enroll2.SPEAKERS_DIR", tmp_path):
            duration = await simulate_enroll_nonblocking(3)
            assert duration >= 0.0

            # Should have created 3 files
            files = list(tmp_path.glob("*.npy"))
            assert len(files) == 3

@pytest.mark.asyncio
async def test_measure_event_loop_lag():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch("benchmark_enroll2.SPEAKERS_DIR", tmp_path):
            duration, max_delay = await measure_event_loop_lag(
                simulate_enroll_nonblocking, 2
            )
            assert duration >= 0.0
            assert max_delay >= 0.0
