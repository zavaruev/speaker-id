import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from benchmark_enroll import (
    save_embedding_blocking,
    save_embedding_nonblocking,
    simulate_enroll_blocking,
    simulate_enroll_nonblocking,
    measure_event_loop_lag,
)


def test_save_embedding_blocking():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("benchmark_enroll.SPEAKERS_DIR", Path(tmpdir)):
            avg_embeddings_numpy = np.random.rand(10, 10)
            user_id = "test_user_blocking"
            save_embedding_blocking(avg_embeddings_numpy, user_id)

            file_path = os.path.join(tmpdir, f"{user_id}.npy")
            assert os.path.exists(file_path)
            # Verify it's actually a valid numpy file we can load
            loaded = np.load(file_path)
            np.testing.assert_array_equal(loaded, avg_embeddings_numpy)


def test_save_embedding_nonblocking():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("benchmark_enroll.SPEAKERS_DIR", Path(tmpdir)):
            avg_embeddings_numpy = np.random.rand(10, 10)
            user_id = "test_user_nonblocking"
            save_embedding_nonblocking(avg_embeddings_numpy, user_id)

            file_path = os.path.join(tmpdir, f"{user_id}.npy")
            assert os.path.exists(file_path)
            loaded = np.load(file_path)
            np.testing.assert_array_equal(loaded, avg_embeddings_numpy)


@pytest.mark.asyncio
async def test_simulate_enroll_blocking():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("benchmark_enroll.SPEAKERS_DIR", Path(tmpdir)):
            duration = await simulate_enroll_blocking(num_requests=2)
            assert duration >= 0.0


@pytest.mark.asyncio
async def test_simulate_enroll_nonblocking():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("benchmark_enroll.SPEAKERS_DIR", Path(tmpdir)):
            duration = await simulate_enroll_nonblocking(num_requests=2)
            assert duration >= 0.0


@pytest.mark.asyncio
async def test_measure_event_loop_lag():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("benchmark_enroll.SPEAKERS_DIR", Path(tmpdir)):
            duration, max_delay = await measure_event_loop_lag(
                simulate_enroll_nonblocking, num_requests=2
            )
            assert duration >= 0.0
            assert max_delay >= 0.0
