import asyncio
import time
import subprocess
from fastapi.concurrency import run_in_threadpool

async def measure_event_loop_lag(coro_func):
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
    start_time = time.perf_counter()
    await coro_func()
    duration = time.perf_counter() - start_time
    keep_running = False
    await ticker_task

    return duration, max_delay

def run_blocking():
    subprocess.run(["sleep", "1"], check=True)

async def run_blocking_in_loop():
    tasks = [asyncio.to_thread(run_blocking) for _ in range(5)]
    await asyncio.gather(*tasks)

async def run_async_subprocess():
    async def run_one():
        proc = await asyncio.create_subprocess_exec("sleep", "1")
        await proc.communicate()
    tasks = [run_one() for _ in range(5)]
    await asyncio.gather(*tasks)

async def run_subprocess_run():
    def run_one():
        subprocess.run(["sleep", "1"])
    async def do_run():
        run_one()
    tasks = [do_run() for _ in range(5)]
    await asyncio.gather(*tasks)

async def main():
    print("Running blocking subprocess in event loop directly...")
    try:
         d, l = await measure_event_loop_lag(run_subprocess_run)
         print(f"Duration: {d:.4f}, max lag: {l:.4f}")
    except Exception as e:
         print(e)

    print("\nRunning run_in_threadpool...")
    d, l = await measure_event_loop_lag(run_blocking_in_loop)
    print(f"Duration: {d:.4f}, max lag: {l:.4f}")

    print("\nRunning async subprocess...")
    d, l = await measure_event_loop_lag(run_async_subprocess)
    print(f"Duration: {d:.4f}, max lag: {l:.4f}")

asyncio.run(main())
