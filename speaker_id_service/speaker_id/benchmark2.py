import asyncio
import time
import os
import tempfile
from fastapi.concurrency import run_in_threadpool

def create_large_file(path, size_mb=10):
    with open(path, "wb") as f:
        f.write(os.urandom(size_mb * 1024 * 1024))

async def process_blocking(file_path: str):
    os.remove(file_path)

async def simulate_concurrent_requests_blocking(num_requests):
    file_paths = []
    for _ in range(num_requests):
        path = tempfile.mktemp()
        create_large_file(path, size_mb=50) # 50MB file
        file_paths.append(path)

    start_time = time.perf_counter()
    tasks = [process_blocking(path) for path in file_paths]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start_time

async def process_nonblocking(file_path: str):
    await run_in_threadpool(os.remove, file_path)

async def simulate_concurrent_requests_nonblocking(num_requests):
    file_paths = []
    for _ in range(num_requests):
        path = tempfile.mktemp()
        create_large_file(path, size_mb=50) # 50MB file
        file_paths.append(path)

    start_time = time.perf_counter()
    tasks = [process_nonblocking(path) for path in file_paths]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start_time


async def measure_event_loop_lag(coro_func, num_requests):
    # We run a background task that ticks every 10ms and measures the max delay
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
    num_requests = 100

    print(f"\nSimulating {num_requests} concurrent requests (50MB file each) for os.remove...")

    blocking_duration, blocking_lag = await measure_event_loop_lag(simulate_concurrent_requests_blocking, num_requests)
    print(f"Blocking method (os.remove): {blocking_duration:.4f}s total time, max event loop delay: {blocking_lag:.4f}s")


    nonblocking_duration, nonblocking_lag = await measure_event_loop_lag(simulate_concurrent_requests_nonblocking, num_requests)
    print(f"Non-blocking method (run_in_threadpool(os.remove)): {nonblocking_duration:.4f}s total time, max event loop delay: {nonblocking_lag:.4f}s")


if __name__ == "__main__":
    asyncio.run(main())
