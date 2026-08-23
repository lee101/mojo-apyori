"""Benchmark mojo-apyori against upstream apyori on identical transactions."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import pathlib
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import apyori  # noqa: E402


def load_upstream():
    distribution = importlib.metadata.distribution("apyori")
    source = pathlib.Path(distribution.locate_file("apyori.py"))
    spec = importlib.util.spec_from_file_location("upstream_apyori_bench", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


upstream = load_upstream()


def timeit(function, repeat=3):
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def transactions(seed, rows, items, width):
    rng = np.random.default_rng(seed)
    return [
        sorted(rng.choice(items, size=width, replace=False).tolist())
        for _ in range(rows)
    ]


def machine():
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return model or platform.machine()


def main():
    cases = [
        (
            "250 baskets, 24 items, width 4, max 2",
            transactions(0, 250, 24, 4),
            {"min_support": 0.02, "max_length": 2},
        ),
        (
            "10k baskets, 8 items, width 2, max 1",
            transactions(3, 10_000, 8, 2),
            {"min_support": 0.02, "max_length": 1},
        ),
        (
            "25k baskets, 48 items, width 10, max 3",
            transactions(1, 25_000, 48, 10),
            {"min_support": 0.02, "max_length": 3},
        ),
        (
            "60k baskets, 100 items, width 5, max 2",
            transactions(2, 60_000, 100, 5),
            {"min_support": 0.001, "max_length": 2},
        ),
    ]
    list(apyori.apriori([["a", "b"], ["a"]], min_support=0.1))
    print(f"Machine: {machine()} ({os.cpu_count()} logical CPUs)")
    print()
    print("| workload | mojo-apyori | upstream apyori | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, rows, kwargs in cases:
        mojo_time, mojo_result = timeit(
            lambda: list(apyori.apriori(rows, **kwargs))
        )
        python_time, python_result = timeit(
            lambda: list(upstream.apriori(rows, **kwargs))
        )
        assert mojo_result == python_result
        ratio = python_time / mojo_time
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1000:.1f} ms | "
            f"{python_time * 1000:.1f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
