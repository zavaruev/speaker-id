import asyncio
import time
import numpy as np
import shutil
import uuid
import tempfile
import os
from pathlib import Path
from fastapi.concurrency import run_in_threadpool

# Mock constants
SPEAKERS_DIR = Path("/tmp/speakers_mock")
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

def mock_rebuild_cache():
    pass

def save_embedding(avg_embeddings_numpy, user_id):
    tmp_save = f"/tmp/.{uuid.uuid4()}"
    np.save(tmp_save, avg_embeddings_numpy)
    shutil.move(tmp_save + ".npy", str(SPEAKERS_DIR / f"{user_id}.npy"))
    mock_rebuild_cache()

async def simulate_enroll_blocking(num_requests):
    async def process_blocking(user_id):
        avg_embeddings_numpy = np.random.rand(512) # Realistic size for embedding
        # Blocking I/O
        save_embedding(avg_embeddings_numpy, user_id)
        # Simulate some network delay or other await to yield
        await asyncio.sleep(0.05)

    start_time = time.perf_counter()
    tasks = [process_blocking(f"user_{i}") for i in range(num_requests)]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start_time

async def simulate_enroll_nonblocking(num_requests):
    async def process_nonblocking(user_id):
        avg_embeddings_numpy = np.random.rand(512)
        # Non-blocking I/O
        await run_in_threadpool(save_embedding, avg_embeddings_numpy, user_id)
        await asyncio.sleep(0.05)

    start_time = time.perf_counter()
    tasks = [process_nonblocking(f"user_{i}") for i in range(num_requests)]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start_time

async def measure_event_loop_lag(coro_func, num_requests):
    max_delay = 0
    keep_running = True

    async def ticker():
        nonlocal max_delay
        while keep_running:
            start = time.perf_counter()
            await asyncio.sleep(0.01)
            delay = time.perf_counter() - start - 0.01
            if delay > max_delay:
                max_delay = delay

    ticker_task = asyncio.create_task(ticker())
    duration = await coro_func(num_requests)
    keep_running = False
    await ticker_task

    return duration, max_delay

async def main():
    print("Simulating concurrent enrollments (1000 requests, 512-dim embedding)...")
    num_requests = 1000

    # Warmup
    await simulate_enroll_nonblocking(10)

    # Run blocking
    blocking_duration, blocking_lag = await measure_event_loop_lag(simulate_enroll_blocking, num_requests)
    print(f"Blocking method: {blocking_duration:.4f}s total time, max event loop delay: {blocking_lag:.4f}s")

    # Run non-blocking
    nonblocking_duration, nonblocking_lag = await measure_event_loop_lag(simulate_enroll_nonblocking, num_requests)
    print(f"Non-blocking method: {nonblocking_duration:.4f}s total time, max event loop delay: {nonblocking_lag:.4f}s")

    # Cleanup
    shutil.rmtree(SPEAKERS_DIR)

if __name__ == "__main__":
    asyncio.run(main())
