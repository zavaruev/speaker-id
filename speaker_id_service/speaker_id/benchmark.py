import asyncio
import time
import shutil
import tempfile
from fastapi import UploadFile
from starlette.datastructures import Headers
import os
from fastapi.concurrency import run_in_threadpool

def create_large_file(path, size_mb=10):
    with open(path, "wb") as f:
        f.write(os.urandom(size_mb * 1024 * 1024))

async def simulate_concurrent_requests_blocking(upload_files):
    async def process_blocking(file: UploadFile):
        temp_input = tempfile.mktemp()
        # Blocking I/O
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.seek(0)
        os.remove(temp_input)

    start_time = time.perf_counter()
    tasks = [process_blocking(file) for file in upload_files]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start_time

async def simulate_concurrent_requests_nonblocking(upload_files):
    def save_file(src, dest):
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(src, buffer)

    async def process_nonblocking(file: UploadFile):
        temp_input = tempfile.mktemp()
        # Non-blocking I/O
        await run_in_threadpool(save_file, file.file, temp_input)
        file.file.seek(0)
        os.remove(temp_input)

    start_time = time.perf_counter()
    tasks = [process_nonblocking(file) for file in upload_files]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start_time

async def measure_event_loop_lag(coro_func, upload_files):
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

    duration = await coro_func(upload_files)

    keep_running = False
    await ticker_task

    return duration, max_delay

async def main():
    print("Preparing test files...")
    dummy_path = "dummy_audio.wav"
    create_large_file(dummy_path, size_mb=20) # 20MB file

    num_requests = 10
    upload_files = []
    for _ in range(num_requests):
        f = open(dummy_path, "rb")
        upload_files.append(UploadFile(filename="test.wav", file=f, headers=Headers()))

    print(f"\nSimulating {num_requests} concurrent requests (20MB file each)...")

    # Run non-blocking first to warm up any caches
    _, _ = await measure_event_loop_lag(simulate_concurrent_requests_nonblocking, upload_files)
    for u in upload_files:
        u.file.seek(0)

    blocking_duration, blocking_lag = await measure_event_loop_lag(simulate_concurrent_requests_blocking, upload_files)
    print(f"Blocking method: {blocking_duration:.4f}s total time, max event loop delay: {blocking_lag:.4f}s")

    for u in upload_files:
        u.file.seek(0)

    nonblocking_duration, nonblocking_lag = await measure_event_loop_lag(simulate_concurrent_requests_nonblocking, upload_files)
    print(f"Non-blocking method: {nonblocking_duration:.4f}s total time, max event loop delay: {nonblocking_lag:.4f}s")

    for u in upload_files:
        u.file.close()
    os.remove(dummy_path)

if __name__ == "__main__":
    asyncio.run(main())
