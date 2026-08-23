"""ctypes bindings for the Mojo Apriori kernels."""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(
    os.environ.get("MOJO_APYORI_LIB", ROOT / "dist" / "libmojo-apyori.so")
)
I = ctypes.c_int64
_I64_MAX = np.iinfo(np.int64).max
_PARALLEL_CANDIDATES = 256
_PARALLEL_BITMAP_WORK = 1 << 17
_PARALLEL_SPARSE_WORK = 1 << 20
_PARALLEL_WORKERS = min(8, os.cpu_count() or 1)

_SIGNATURES = {
    "map_build_bitmaps": ([I] * 5, None),
    "map_count_bitmap": ([I] * 6, None),
    "map_count_sparse": ([I] * 6, None),
}

_library: ctypes.CDLL | None = None
_executor: ThreadPoolExecutor | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIB.exists():
            raise RuntimeError("Mojo library not built; run `pixi run build`")
        _library = ctypes.CDLL(str(LIB))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS)
    return _executor


def _ranges(size: int) -> list[tuple[int, int]]:
    workers = min(_PARALLEL_WORKERS, size)
    chunk = (size + workers - 1) // workers
    return [
        (start, min(start + chunk, size))
        for start in range(0, size, chunk)
    ]


def _array(
    array: np.ndarray,
    *,
    dtype: np.dtype,
    name: str,
    writable: bool = False,
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError(f"{name} must be non-empty and non-null")
    return array


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.uint64)):
        raise TypeError("FFI buffers must use int64 or uint64 elements")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)


def _dimension(value: int, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > _I64_MAX:
        raise ValueError(f"{name} is outside the native int64 range")
    return value


def build_bitmaps(
    tids: np.ndarray,
    offsets: np.ndarray,
    item_count: int,
    words: int,
    bitmaps: np.ndarray,
) -> None:
    """Validate and retain all buffers for the duration of the native call."""
    item_count = _dimension(item_count, "item_count")
    words = _dimension(words, "words")
    tids = _array(tids, dtype=np.dtype(np.int64), name="tids")
    offsets = _array(offsets, dtype=np.dtype(np.int64), name="offsets")
    bitmaps = _array(
        bitmaps, dtype=np.dtype(np.uint64), name="bitmaps", writable=True
    )
    if offsets.size != item_count + 1:
        raise ValueError("offsets length must equal item_count + 1")
    if bitmaps.size != item_count * words:
        raise ValueError("bitmaps size does not match item_count * words")
    if offsets[0] != 0 or offsets[-1] != tids.size:
        raise ValueError("offsets must span the tids buffer")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be nondecreasing")
    if np.any(tids < 0) or np.any(tids >= words * 64):
        raise ValueError("transaction IDs exceed the bitmap bounds")
    lib().map_build_bitmaps(
        addr(tids), addr(offsets), item_count, words, addr(bitmaps)
    )


def count_bitmap(
    candidates: np.ndarray,
    candidate_count: int,
    candidate_length: int,
    bitmaps: np.ndarray,
    words: int,
    counts: np.ndarray,
) -> None:
    candidate_count = _dimension(candidate_count, "candidate_count")
    candidate_length = _dimension(candidate_length, "candidate_length")
    words = _dimension(words, "words")
    candidates = _array(
        candidates, dtype=np.dtype(np.int64), name="candidates"
    )
    bitmaps = _array(bitmaps, dtype=np.dtype(np.uint64), name="bitmaps")
    counts = _array(
        counts, dtype=np.dtype(np.int64), name="counts", writable=True
    )
    if candidate_count == 0 or candidate_length == 0 or words == 0:
        raise ValueError("native count dimensions must be positive")
    if candidates.size != candidate_count * candidate_length:
        raise ValueError("candidates size does not match its dimensions")
    if counts.size != candidate_count or bitmaps.size % words:
        raise ValueError("output or bitmap size does not match its dimensions")
    item_count = bitmaps.size // words
    if np.any(candidates < 0) or np.any(candidates >= item_count):
        raise ValueError("candidate item ID exceeds the bitmap bounds")
    function = lib().map_count_bitmap
    work = candidate_count * candidate_length * words
    if (
        candidate_count < _PARALLEL_CANDIDATES
        or work < _PARALLEL_BITMAP_WORK
        or _PARALLEL_WORKERS == 1
    ):
        function(
            addr(candidates),
            candidate_count,
            candidate_length,
            addr(bitmaps),
            words,
            addr(counts),
        )
        return

    def run(part):
        start, stop = part
        function(
            addr(candidates[start * candidate_length :]),
            stop - start,
            candidate_length,
            addr(bitmaps),
            words,
            addr(counts[start:]),
        )

    list(_pool().map(run, _ranges(candidate_count)))


def count_sparse(
    candidates: np.ndarray,
    candidate_count: int,
    candidate_length: int,
    tids: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
) -> None:
    candidate_count = _dimension(candidate_count, "candidate_count")
    candidate_length = _dimension(candidate_length, "candidate_length")
    candidates = _array(
        candidates, dtype=np.dtype(np.int64), name="candidates"
    )
    tids = _array(tids, dtype=np.dtype(np.int64), name="tids")
    offsets = _array(offsets, dtype=np.dtype(np.int64), name="offsets")
    counts = _array(
        counts, dtype=np.dtype(np.int64), name="counts", writable=True
    )
    if candidate_count == 0 or candidate_length == 0:
        raise ValueError("native count dimensions must be positive")
    if candidates.size != candidate_count * candidate_length:
        raise ValueError("candidates size does not match its dimensions")
    if counts.size != candidate_count or offsets.size < 2:
        raise ValueError("output or offsets size does not match its dimensions")
    if offsets[0] != 0 or offsets[-1] != tids.size:
        raise ValueError("offsets must span the tids buffer")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be nondecreasing")
    item_count = offsets.size - 1
    if np.any(candidates < 0) or np.any(candidates >= item_count):
        raise ValueError("candidate item ID exceeds the sparse index bounds")
    function = lib().map_count_sparse
    work = candidate_count * candidate_length * tids.size
    if (
        candidate_count < _PARALLEL_CANDIDATES
        or work < _PARALLEL_SPARSE_WORK
        or _PARALLEL_WORKERS == 1
    ):
        function(
            addr(candidates),
            candidate_count,
            candidate_length,
            addr(tids),
            addr(offsets),
            addr(counts),
        )
        return

    def run(part):
        start, stop = part
        function(
            addr(candidates[start * candidate_length :]),
            stop - start,
            candidate_length,
            addr(tids),
            addr(offsets),
            addr(counts[start:]),
        )

    list(_pool().map(run, _ranges(candidate_count)))
