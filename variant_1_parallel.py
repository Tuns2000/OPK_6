"""Параллельные реализации практического занятия 6 для варианта 1.

Исходный ``variant_1.py`` намеренно не изменяется: все дополнительные модели
выполнения и измерения находятся в этом файле.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import multiprocessing
import os
import platform
import statistics
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from variant_1 import (
    SCALES,
    cpu_job,
    io_job,
    io_job_async,
    make_workload,
    run_sequential,
)


def _cpu_workers(task_count: int) -> int:
    return max(1, min(os.cpu_count() or 1, task_count))


def _io_workers(task_count: int) -> int:
    return max(1, min(32, task_count))


def _cpu_batch(seeds: Sequence[int]) -> float:
    return sum(cpu_job(seed) for seed in seeds)


def run_threaded(
    cpu_seeds: Sequence[int], io_delays: Sequence[float]
) -> tuple[float, int]:
    """Запускает вычисления и ожидания в потоках."""
    with ThreadPoolExecutor(
        max_workers=max(_cpu_workers(len(cpu_seeds)), _io_workers(len(io_delays)))
    ) as pool:
        cpu_futures = [pool.submit(cpu_job, seed) for seed in cpu_seeds]
        io_futures = [pool.submit(io_job, delay) for delay in io_delays]
        cpu_total = sum(future.result() for future in cpu_futures)
        for future in io_futures:
            future.result()
    return cpu_total, len(io_delays)


def run_processes(
    cpu_seeds: Sequence[int], io_delays: Sequence[float]
) -> tuple[float, int]:
    """Выносит CPU-bound часть в процессы, а ожидания выполняет потоками."""
    with ProcessPoolExecutor(max_workers=_cpu_workers(len(cpu_seeds))) as processes:
        cpu_futures = [processes.submit(cpu_job, seed) for seed in cpu_seeds]
        with ThreadPoolExecutor(max_workers=_io_workers(len(io_delays))) as threads:
            io_futures = [threads.submit(io_job, delay) for delay in io_delays]
            cpu_total = sum(future.result() for future in cpu_futures)
            for future in io_futures:
                future.result()
    return cpu_total, len(io_delays)


async def _run_asyncio(
    cpu_seeds: Sequence[int], io_delays: Sequence[float]
) -> tuple[float, int]:
    loop = asyncio.get_running_loop()
    cpu_future = loop.run_in_executor(None, _cpu_batch, list(cpu_seeds))
    io_tasks = [asyncio.create_task(io_job_async(delay)) for delay in io_delays]
    cpu_total, _ = await asyncio.gather(cpu_future, asyncio.gather(*io_tasks))
    return cpu_total, len(io_delays)


def run_asyncio(
    cpu_seeds: Sequence[int], io_delays: Sequence[float]
) -> tuple[float, int]:
    """Перекрывает asyncio-ожидания с CPU-работой вне event loop."""
    return asyncio.run(_run_asyncio(cpu_seeds, io_delays))


Model = Callable[[Sequence[int], Sequence[float]], tuple[float, int]]
MODELS: dict[str, Model] = {
    "sequential": run_sequential,
    "threads": run_threaded,
    "processes": run_processes,
    "asyncio": run_asyncio,
}


@dataclass(frozen=True)
class Timing:
    median: float
    minimum: float
    maximum: float


def measure(
    model: Model,
    cpu_seeds: Sequence[int],
    io_delays: Sequence[float],
    repeats: int = 5,
) -> Timing:
    """Возвращает медиану и min-max, исключая первый прогревочный запуск."""
    if repeats < 1:
        raise ValueError("repeats должен быть положительным")
    model(cpu_seeds, io_delays)
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        model(cpu_seeds, io_delays)
        values.append(time.perf_counter() - started)
    return Timing(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
    )


def workload_profile(
    cpu_seeds: Sequence[int], io_delays: Sequence[float]
) -> tuple[float, float]:
    """Измеряет отдельно вычислительную и ожидающую части эталона."""
    started = time.perf_counter()
    for seed in cpu_seeds:
        cpu_job(seed)
    cpu_time = time.perf_counter() - started

    started = time.perf_counter()
    for delay in io_delays:
        io_job(delay)
    io_time = time.perf_counter() - started
    return cpu_time, io_time


def amdahl_speedup(serial_fraction: float, workers: int) -> float:
    """Теоретическое ускорение по закону Амдала."""
    if not 0 <= serial_fraction <= 1:
        raise ValueError("serial_fraction должен быть в диапазоне [0, 1]")
    if workers < 1:
        raise ValueError("workers должен быть положительным")
    return 1 / (serial_fraction + (1 - serial_fraction) / workers)


def _format_row(name: str, timing: Timing, baseline: float, workers: int) -> str:
    speedup = baseline / timing.median
    efficiency = speedup / workers
    return (
        f"{name:<12} median={timing.median:8.4f}s "
        f"range=[{timing.minimum:.4f}, {timing.maximum:.4f}] "
        f"speedup={speedup:6.2f} efficiency={efficiency:6.2f}"
    )


def run_report(scales: Iterable[int], repeats: int = 5) -> None:
    print(f"Python: {platform.python_version()} | OS: {platform.platform()}")
    print(f"CPU: {os.cpu_count() or 1} logical cores")
    for scale in scales:
        cpu_seeds, io_delays = make_workload(scale)
        cpu_time, io_time = workload_profile(cpu_seeds, io_delays)
        total = cpu_time + io_time
        print(f"\nscale={scale}, cpu share={cpu_time / total:.1%}, io share={io_time / total:.1%}")
        print("Amdahl upper bounds (f=0: задания независимы, последовательных этапов нет):")
        print("  " + ", ".join(
            f"p={workers}: {amdahl_speedup(0.0, workers):.2f}x"
            for workers in (2, 4, 8)
        ))
        timings = {
            name: measure(model, cpu_seeds, io_delays, repeats)
            for name, model in MODELS.items()
        }
        baseline = timings["sequential"].median
        worker_counts = {
            "sequential": 1,
            "threads": max(_cpu_workers(len(cpu_seeds)), _io_workers(len(io_delays))),
            "processes": _cpu_workers(len(cpu_seeds)) + _io_workers(len(io_delays)),
            "asyncio": len(io_delays) + 1,
        }
        for name, timing in timings.items():
            print(_format_row(name, timing, baseline, worker_counts[name]))


def check_correctness() -> None:
    for scale in SCALES:
        cpu_seeds, io_delays = make_workload(scale)
        expected = run_sequential(cpu_seeds, io_delays)
        for name, model in MODELS.items():
            actual = model(cpu_seeds, io_delays)
            assert math.isclose(actual[0], expected[0], rel_tol=1e-12)
            assert actual[1] == expected[1], name
    assert math.isclose(amdahl_speedup(0.5, 2), 1.3333333333, rel_tol=1e-9)
    print("Correctness checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="проверить совпадение результатов")
    parser.add_argument("--repeats", type=int, default=5, help="число измерений после прогрева")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()

    if args.check:
        check_correctness()
    else:
        run_report(args.scales, args.repeats)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
