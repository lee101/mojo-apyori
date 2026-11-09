# ANALYZE — mojo-apyori hotspot map (measured, not estimated)

Date: 2026-11-09 session. Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical CPUs
(measured `/proc/cpuinfo`; `flags` include AVX/AVX2/popcnt/sse4, SIMD width 4 — verified
by compiling `simd_width_of[DType.uint64]()` and `[DType.float64]()`), 264 GB RAM,
NVIDIA GeForce RTX 5090 32 GB / compute 12.0 present but unused by the port.
Toolchain: `mojo 1.2.0.dev2026092605`, numpy 2.5.1, Python 3.13 (pixi env),
upstream apyori 1.1.2 installed as a pypi dependency and importable
(`.pixi/envs/default/lib/python3.13/site-packages/apyori.py`).

Every number below comes from a run executed in this session. No algorithm was changed.
Upstream comparisons use `bench/bench.py`'s own loader (`spec_from_file_location`)
so the two modules coexist by name.

---

## 1. End-to-end (`pixi run bench`, flock-serialized) — reproduced in this session

```
Pixi task (bench): flock /tmp/mojo-bench.lock python bench/bench.py
Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (72 logical CPUs)

| workload | mojo-apyori | upstream apyori | result |
| --- | ---: | ---: | ---: |
| 250 baskets, 24 items, width 4, max 2 | 2.5 ms | 2.8 ms | 1.13x faster |
| 10k baskets, 8 items, width 2, max 1 | 4.0 ms | 4.7 ms | 1.15x faster |
| 25k baskets, 48 items, width 10, max 3 | 162.8 ms | 4198.9 ms | 25.80x faster |
| 60k baskets, 100 items, width 5, max 2 | 172.1 ms | 1102.1 ms | 6.40x faster |
```

Decomposition of the same four workloads (median of 11, min-based; run `/tmp/analyze_bench.py §A`):

| workload | `apriori` median | upstream median | ingestion `TransactionManager()` (mojo / upstream) | `gen_support_records` alone (prebuilt TM, warm) | counted support records |
| --- | ---: | ---: | ---: | ---: | --- |
| 250x24 w4 max2 | 2.33 ms | 2.76 ms | 0.13 / 0.16 ms | 0.46 ms | 200 |
| 10k x8 w2 max1 | 4.36 ms | 4.88 ms | 3.66 / 4.44 ms | 0.013 ms | 8 |
| 25k x48 w10 max3 | 176.3 ms | 4307 ms | 47.9 / 55.7 ms | 52.8 ms | 1176 |
| 60k x100 w5 max2 | 169.5 ms | 1179 ms | 73.4 / 98.0 ms | 12.2 ms | 5050 |

Reading this table: the port wins everywhere on this machine, but the wins are
structurally different. At 250–10k rows the kernel is idle and the wall is ingestion
plus ordered statistics; at 25k–60k rows the kernel (or its Python encoding) dominates.

---

## 2. Ranked hotspot table

Rank is by observed wall time at the largest real workload (60k) plus systemic cost
across all four. `bytes touched` / `flops` are computed from the measured shapes in §4
(e.g. `count_bitmap` bytes = `n·k·words·8`), never guessed.

| rank | function | input size | time (median, this session) | bytes touched | flops | flops/byte | bound | Python/ctypes glue or kernel |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `TransactionManager.__init__` ingestion (`python/apyori/__init__.py:38`) | 60k rows × width 5 = 300k item refs; 100 items | 73–88 ms wall (0.25 µs/ref avg over 300k; cProfile 0.152 s tottime for 25k build incl. `set.add`/`dict.get`) | Python objects: 300k hash+set ops dominate; numpy touched ≈ 0 at this stage | — (hash/set, not flops) | — | latency (Python interpreter) | Python glue (no kernel involved) |
| 2 | `create_next_candidates` L=3 subset-prune (`__init__.py:184`) | 25k workload: prev = 1128 frequent L2, items = 48 → enumerates C(48,3) = 17296 | 47.2 ms median (set input — the real `gen_support_records` path) / 987 ms if prev were a list | Python: ~17k frozensets + 3 membership probes each; bytes ≈ O(candidates × set overhead), not flat arrays | — | — | latency / Python combinatorics | Python glue |
| 3 | `_count_candidates` candidate encoding (dict lookups, `__init__.py:157`) | 25k L3: 17296 × 3 = 51888 codes | 8.18 ms median (5.92 ms min), inside `_count_candidates` warm total 10.6 ms vs raw kernel 3.96 ms | encoded int64 = 51888×8 = 415 KB read + write; plus per-candidate Python iteration | 51,888 dict lookups | — | latency | Python glue |
| 4 | `count_bitmap` kernel (`src/apriori.mojo:196`, SIMD W=4) | 25k L3: n=17296, k=3, words=391 | 3.96 ms (wrapper, parallel) / 10.6 ms (serial raw) | 17296×3×391×8 = 162 MB (+0.14 MB counts) | 17296×391×3 = 20.3M | 0.125 | memory-bound | kernel |
| 5 | `count_bitmap` kernel | 60k L2: n=4950, k=2, words=938 | 1.12 ms (wrapper, parallel) / 5.33 ms (serial raw) | 4950×2×938×8 = 74.3 MB | 4950×938×2 = 9.3M | 0.125 | memory-bound | kernel |
| 6 | `gen_ordered_statistics` (`__init__.py:234`) | 60k: 5050 records → 14950 `calc_support` calls (2^k−1 per record) | 38.2 ms median (mojo) vs 519 ms (upstream) | support-cache lookups; cache-hit path ≈ 0.6–0.7 µs/call | ~14950 dict hits + frozenset diffs | — | latency | Python glue (reuse of native supports is why mojo wins 13.6×) |
| 7 | `_native_index` (`__init__.py:110`) — `tids` fromiter + `np.zeros` bitmaps + `build_bitmaps` | 25k: tids 250k×8 = 2 MB; bitmaps 48×391×8 = 150 KB | ~10.9 ms of the 59 ms cold build; `build_bitmaps` alone 1.51 ms; 60k native-only ≈ 17.9 ms | 2 MB + 0.15 MB (+ zeroing 60k bitmaps 750 KB) | `build_bitmaps`: 250k inserts = 250k ORs+shifts | 0.125 | memory / latency mix | glue + kernel (one call each) |
| 8 | `count_bitmap` validation scans (`_lib.py:123–157`, `np.any`) | L3 n=51888 | subset-only measurement 0.108 ms/call (all scans ≈ 0.1–0.3 ms/call) | candidates 415 KB × 2 scans + offsets × 2 scans | — | — | memory (streaming) | Python/ctypes glue |
| 9 | `gen_support_records` per-level filter (`__init__.py:220`) | 25k L3: 17296 iterations of Python `zip` + `float(count)/denom` + `frozenset()` + cache store | whole `gen_support_records` 52.8 ms at 25k; filter loop is the non-count remainder after `_count_candidates` ≈ 14.7 ms | 17296 Python objects | — | — | latency | Python glue |
| 10 | `count_sparse` kernel (`src/apriori.mojo:219`) | forced call, 25k k=2 n=2000 (sorted tids) | 120 ms (66–97× slower than bitmap, sorted-tids; 58 ms on unsorted tids) | irregular: pivot stream ≈ 2000×5208×8 = 83 MB + binary-search probes ≈ 2000×13×8 = 0.2 MB scattered | ≈ 2000×5208×log ≈ 133M comparisons | ~1.6 nominal, effectively latency | latency | kernel (when invoked) |
| — | `calc_support` Python fallback (`__init__.py:70`) | never on this machine at bench sizes (only L1 path bypasses kernel; `gen_ordered_statistics` hits cache) | 0.6 µs cached; 1.0 µs uncached 3-item intersection | set intersection over ~3k-element sets | — | — | latency | Python glue |
| — | `parse_args` / `load_transactions` / `dump_as_json` / `main` | CLI only | `dump_as_json` 0.050 ms/record; `apriori` never touches them | — | — | — | — | Python glue (not on bench path) |

GPU verdict rows (flops/byte behind the verdict, §5): `count_bitmap` 0.125, `build_bitmaps` 0.125,
`count_sparse` ≈ irregular/latency — all three are `none` for GPU.

---

## 3. Per-workload wall attribution (measured stage splits)

**25k × 48 w10 max3** (the most kernel-important workload). Stage timers in this session:

| stage | time | note |
| --- | --- | --- |
| ingestion | 77.1 ms | largest single stage even here |
| `_native_index` (cold) | 18.2 ms | of which `build_bitmaps` 1.5 ms; rest is `sorted`/offsets/`fromiter` Python |
| `_count_candidates` L1 (n=48) | 0.07 ms | |
| `create_next_candidates` L2 | 0.41 ms | |
| `_count_candidates` L2 (n=1128, k=2, words=391) | 2.82 ms | wrapper-parallel; serial raw 0.37 ms — wrapper is **0.41× slower** here |
| `create_next_candidates` L3 (from set) | 66.4 ms | list variant: 987 ms (20.9× worse; upstream identical) |
| `_count_candidates` L3 (n=17296, k=3) | 11.8 ms | encoding ≈ 8.2 ms + kernel ≈ 3.96 ms |
| `gen_ordered_statistics` (1176 records, 3432 calls) | 7.07 ms (mojo) vs 219 ms (upstream) | |

**60k × 100 w5 max2:**

| stage | time | note |
| --- | --- | --- |
| ingestion | 91.6 ms | |
| `_native_index` (cold) | 19.7 ms | |
| `_count_candidates` L1 (n=100) | 0.03 ms | pure Python `len()` path |
| `create_next_candidates` L2 (n=4950) | 19.8 ms | mojo 1.25 ms isolated vs upstream 1.49 ms |
| `_count_candidates` L2 (n=4950, k=2, words=938) | 7.5 ms | wrapper-parallel 1.12 ms vs serial 5.33 ms (4.77× parallel win) |
| `gen_ordered_statistics` (5050 records, 14950 calls) | 38.2 ms (mojo) vs 519 ms (upstream) | cache reuse is the mojo advantage |

**250 × 24 w4 max2 (smallest):** `apriori` 2.33 ms; stages: ingestion 0.13 ms,
`_count_candidates` 0.31 ms total (L1 0.02 ms + L2 0.29 ms), `gen_ordered_statistics`
1.17 ms over 200 records (552 calls). Kernel calls: 2× `count_bitmap`. The wall is
ordered statistics, not counting.

**10k × 8 w2 max1 (singleton):** `apriori` 4.36 ms; ingestion 3.66 ms (84%); kernel calls: 0
(`length == 1` branch takes pure-Python `len()`). This is the correct design — measured
L1 `_count_candidates` is 5.0 µs. Upstream ingestion is 4.44 ms; the port is faster here
only because the forked `__init__` inlines `add_transaction`.

---

## 4. Kernel microbenchmarks (direct ctypes, warm)

### `count_bitmap` (SIMD W=4 `AND` + `pop_count` → `reduce_add`, scalar tail)

| case | shape | median | bytes/cand | eff. BW | verdict |
| --- | --- | --- | ---: | ---: | --- |
| A tiny | n=276, k=2, words=4 | 0.036 ms | 64 B | — | glue floor |
| B | n=5000, k=3, words=157 | 0.905 ms (0.81 min) | 3768 B | — | glue + small kernel |
| C L2 | n=1128, k=2, words=391 | 0.717 ms wrapper / 0.366 ms serial | 6256 B | — | |
| C L3 | n=5000, k=3, words=391 | 1.078 ms wrapper | 9384 B | 43.5 GB/s @ n=17296 (162 MB / 3.96 ms ≈ 41 GB/s) | memory-bound |
| D L2 | n=4950, k=2, words=938 | 1.345 ms wrapper (synthetic) / 1.12 ms at real batch; serial 5.33 ms | 15008 B | 66 GB/s (74.3 MB / 1.12 ms, parallel) | memory-bound |

flops/byte derivation (same for every bitmap row): per `uint64` word you do
`(k−1)` ANDs + 1 `popcount` ≈ `k` ops over `k·8` bytes → **0.125 flops/byte** at
k=2 and k=3 (60000×938 case: 9.3M flops / 74.3 MB = 0.125; 25000×391 case: 20.3M /
162 MB = 0.125). The number is shape-independent.

### `build_bitmaps`

| shape | median |
| --- | --- |
| 250×24 (1000 tids) | 0.041 ms |
| 10k×8 (20000 tids, words=157) | 0.173 ms (0.12 min) |
| 25k×48 (250000 tids, words=391) | 1.508 ms (0.886 min) |
| 60k×100 (300000 tids, words=938) | 1.169 ms (1.11 min) |

memset of the target (`np.zeros`) costs ≈ the same order as the kernel at large sizes
(150 KB / 750 KB). Both are one-shot per `apriori` call.

### `count_sparse` (binary search into sorted segments)

Forced calls at workload-C/D shapes, **sorted tids** (see §7 — the shipped `_native_index`
leaves tids unsorted on the bitmap path, so unsorted numbers are not correctness-valid):

| case | sparse | bitmap (same shape) | ratio |
| --- | ---: | ---: | ---: |
| C k=2 n=2000 | 120.4 ms | 1.24 ms | 97× slower |
| C k=3 n=2000 | 128.7 ms | 1.70 ms | 76× slower |
| D k=2 n=2000 | 61.0 ms | 0.91 ms | 67× slower |
| D k=3 n=2000 | 56.5 ms | 1.14 ms | 50× slower |

Pivot posting lists are ≈ 5208 entries (25k) / ≈ 3000 (60k); each candidate does
`pivot×(k−1)` binary searches. This kernel is latency-bound, not throughput-bound.

### L1 (no kernel)

`_count_candidates` with `length==1` never touches ctypes: pure-Python `len()` of the
index map; measured 5.0 µs for n=8. That is why the 10k workload is ingestion-bound, not kernel-bound.

---

## 5. Numpy and Python baselines

| comparison | numpy/plain-Python time | mojo | ratio |
| --- | --- | --- | --- |
| bitmap AND+popcount, C k=2 n=2000 words=391 (`take` + byte table) | 32.2 ms | 0.636 ms | 50.7× faster (mojo) |
| bitmap, C k=3 n=5000 | 84.1 ms | 1.99 ms | 42.2× |
| bitmap, D k=2 n=2000 words=938 | 69.3 ms | 1.77 ms | 39.3× |
| bitmap, D k=3 n=5000 | 245.0 ms | 4.77 ms | 51.4× |
| bitmap, tiny k=2/k=3 | 0.41 / 1.12 ms | 0.06 / 0.09 ms | 6.8–12.7× |
| bool matrix (`minimum.reduce` over item×txn), C k=2 n=2000 rows=25000 | 155.6 ms | 0.9 ms | 177× |
| bool matrix, C k=3 n=3000 | 315.6 ms | 1.7 ms | 189× |
| bool matrix, D k=2 n=2000 rows=60000 | 374.3 ms | 1.7 ms | 214× |
| bool matrix, D k=3 n=3000 | 723.1 ms | 2.2 ms | 326× |
| `create_next_candidates` L2 (n=24→276, n=48→1128, n=100→4950) mojo vs upstream | 0.062 vs 0.070 ms; (25k L2: 0.219 vs ~0.22 ms); 1.25 vs 1.49 ms | — | parity (source-identical algorithm; tiny mojo lead) |
| `create_next_candidates` L3 (prev from set) mojo vs upstream | 47.2 vs 42.6 ms (set); 987 vs 987 ms (list!) | — | parity; input type matters 20.9× |
| `gen_ordered_statistics` mojo vs upstream | 0.95 vs 1.42 ms (250×24); 7.07 vs 219 ms (25k, 31×); 38.2 vs 519 ms (60k, 13.6×) | — | mojo faster via `_support_cache` reuse of native counts |
| upstream `apriori` vs mojo `apriori` | — | — | 1.13–1.15× at small, 6.4× at 60k, 25.8× at 25k (bench table §1) |

**Where the port is slower than upstream (kept, not dropped):**

1. **No workload in `bench/` is slower end-to-end.** But two kernel regimes are slower
   than a serial equivalent and one than upstream at the margin (§6):
   - `count_bitmap` wrapper at C-L2 (n=1128, work=882k) is **0.41× of serial** — the
     ThreadPool path turns a 0.366 ms kernel into 0.884 ms. At n=256–2048 k=2
     speedup was 0.11–0.69× in the sweep; only at n≥5000 did the wrapper win (1.19× →
     2.18× at n=17296, 4.77× at D-L2). The pool knob is mis-tuned: `pool.map`
     dispatch alone is 0.201 ms per call (n=3 chunks).
   - `/tmp` microbench repeats warm the pool differently than `bench/`'s cold
     single-shot `timeit`; absolute ms differ by ~10% between runs. The ordering
     (pool hurts below ~2000 candidates) held in every run.
   - `create_next_candidates` L3 from a **list** is 987 ms for both — 20.9× worse than
     the set path. `gen_support_records` passes a `set` so the shipped path is fine,
     but any external caller passing a list silently takes the 20× path. Parity
     with upstream here means inheriting upstream's cliff.

2. **Estimates I did not make:** `create_next_candidates` at C-L3 was 780 ms in the
   first session run (list input) and 47 ms on the set path. I report both and
   would be wrong reporting one. Per-call `np.any` validation cost was measured
   as a subset (0.108 ms) — the full count is bounded by 0.3 ms, not subtracted precisely.

---

## 6. Python/ctypes glue audit

**Per-element ctypes re-entry: NONE in the shipped path.** Measured calls per `apriori`
(monkeypatched `_lib` counters):

| workload | `count_bitmap` | `count_sparse` | `build_bitmaps` |
| --- | --- | --- | --- |
| 250×24 | 1 | 0 | 1 |
| 25k | 2 | 0 | 1 |
| 60k | 1 | 0 | 1 |
| 10k | 0 (L1 bypass) | 0 | 0 |

Each level does exactly one batched ctypes call over the whole candidate array.
The hypothetical per-element variant measured for the map:
at n=200, k=2, words=391 the batched raw call is 0.079 ms vs per-element raw ctypes
3.00 ms (**37.8×**) and per-element through the wrapper 7.67 ms (**96.6×**) — so the
batched design is load-bearing. Hot paths that *do* touch Python per element,
without ctypes: candidate encoding `codes[item]` (51,888 lookups, 8.2 ms), the
`gen_support_records` filter zip (17,296 iterations), `frozenset` construction in
`create_next_candidates`, and `TID` set adds in `__init__`.

**Tiny-call floor (ctypes + validation):** n=2, k=2, words=4: wrapper median 32.1 µs
(min 29.9 µs), raw ctypes (no validation) 13.2 µs (min 12.2 µs) → validation ≈ 18.9 µs/call.
`mojo build` notes say ctypes calls are ~0.9 µs; measured floor here is 13 µs raw /
32 µs validated — the extra ~12–30 µs is `addr()` checks, `np.any` scans, and
`_array` guard re-validation, not the ABI.

**Per-call allocations/copies (counted from source + measured arrays):**

| site | allocation | size at bench | per call |
| --- | --- | --- | --- |
| `_native_index`: `offsets` | `np.empty(n_items+1)` | 100×8 = 800 B | 1× per `apriori` |
| `_native_index`: `tids` via `np.fromiter(chain.from_iterable(...))` | fills 2–2.4 MB (25k/60k) | 250k×8 / 300k×8 | 1× per `apriori` |
| `_native_index`: `bitmaps` `np.zeros` (+ memset time) | 150 KB / 750 KB | 1× per `apriori` |
| `_count_candidates`: `encoded` via `np.fromiter(genexpr)` | 415 KB at 25k L3 (51,888×8) | 1× per level |
| `_count_candidates`: L1 `np.fromiter` | 8×8 = 64 B | 1× per `apriori` |
| `_count_candidates`: `counts` `np.empty` | 138 KB at 25k L3 | 1× per level |
| `create_next_candidates`: `frozenset(chain.from_iterable(...))` + sorted list + `combinations` generator | 17,296 frozensets at C L3 | 1× per level |

`tracemalloc` in this session: `TransactionManager` 25.9 MB peak (25k), `_native_index`
2.15 MB traced (numpy buffers partially untraced). The `tids` array is computed on the
bitmap path by iterating Python `set`s in hash order — the memory pass is fine, but
it is the reason tids are unsorted (see §7).

---

## 7. Correctness-adjacent findings (not algorithmic changes, flagged for the next phases)

1. **Sparse kernel precondition vs builder (verified, not speculated):** on all four bench
   shapes the shipped `_native_index` builds `tids` from unsorted set iteration when
   bitmaps are used (measured: 100/100 segments unsorted at 60k, sample `[49152, 57344,
   49155, ...]`). `count_sparse` requires sorted segments (binary search). Forced
   `count_sparse` on those arrays returned `correct=False` at D and `correct=True` only
   by luck at C for one seed. After sorting each segment the sparse kernel is correct
   but 50–97× slower than bitmap. The two kernels never mix in the shipped path
   (`bitmaps is None` ⇒ tids built from `sorted(...)` ⇒ sorted), so this is not a
   shipped-path bug — but §4's sparse numbers are only meaningful on sorted arrays,
   and anyone flipping the 256 MiB gate must sort tids first.
2. **Parallel threshold vs measured optimum (§5):** `_PARALLEL_BITMAP_WORK = 131072`
   fires at ~14× too little work. At C-L2 (work 882k) serial beats the pool 2.4×;
   the crossover in swept runs is work ≈ 3.9M (n=5000, k=2) at 1.19× and solidly 2×+ at
   work ≈ 20M. Raising the threshold (or dropping the pool for a Mojo `parallelize`
   path) is the cheapest kernel-side change; measured cost of staying: 0.52 ms
   (=0.884−0.366) at C-L2 and up to ~1.5 ms per mid-size call at n=256–1024.
3. **`create_next_candidates` list-vs-set cliff:** `prev_candidates` membership is
   `in` on a list in the L≥3 branch. Internal caller passes a `set` → 47 ms.
   External list callers (tests, upstream parity) pay 987 ms with no warning. Either
   document, or coerce once (`prev_set = set(prev_candidates)`) — 1128-element set
   construction is negligible against 940 ms saved.

---

## 8. Next-phase verdicts per hotspot

| hotspot | contention | next phase | one-line rationale |
| --- | --- | --- | --- |
| `TransactionManager.__init__` ingestion | Python per-ref loop | **none** | Pure Python dict/set; Mojo cannot help without changing the public data model, and upstream parity requires the same objects. (Optional later: keep but fuse with `_native_index` to avoid storing sets and re-iterating them.) |
| `create_next_candidates` L3 subset prune | C(48,3) frozenset algebra | **unroll** (algorithmic rewrite) | Replace per-candidate `combinations(candidate, k−1) ⊂ prev` checks with frequency-encoded array tricks (neighbor masks / int IDs); needs new code, not loop tuning. Coerce list→set first (free 20.9×). |
| `_count_candidates` encoding | 51,888 dict lookups, 8.2 ms | **unroll** | Remap items to IDs once per `apriori` (store `np.array[int64]` codes) and vectorize with `np.take`/fancy indexing; dict-lookup-per-item in Python is the dominant per-level cost after ingestion. |
| `count_bitmap` (C-L2, C-L3, D-L2) | 0.125 flops/byte, 24–69 GB/s eff. | **parallel** | Memory-bound at 0.125 flops/byte; fix the pool threshold first (threshold ≈ 3.9M work), then consider Mojo `parallelize` over candidate ranges. SIMD (W=4) already flat; unrolling beyond W=4 has little left (see D-L2 serial vs parallel: 5.33→1.12 ms is threading, not vector width). |
| `count_bitmap` tiny/small (words ≤ 4, n ≤ 276, 250×24 shape) | ctypes+validation floor 19–32 µs/call | **none** | Already optimal for one call per level; per-call validation (≈19 µs) dwarfs the 0.036 ms kernel. Removing guards would be a safety trade, not an optimization. |
| `build_bitmaps` | 0.125 flops/byte, one-shot | **none** | 1.1–1.5 ms once per `apriori`; `np.zeros` memset costs ≈ kernel. Reducing would mean fused bitmap construction inside `_native_index` — wrapper-level, not kernel-level. |
| `count_sparse` | binary-search latency, irregular | **none** | 50–97× slower than bitmap at every bench shape; only used when bitmaps exceed 256 MiB (never at bench sizes). Leave serial; if ever activated, the fix is early-exit pivot ordering, not threads. |
| `gen_ordered_statistics` / `calc_support` | cache-hit lookups | **none** | Never the wall: 0.6 µs/call × 14950 = 9 ms of dict hits inside 38 ms; the frozenset diffs dominate. The native-support reuse already extracted the win. |
| `gen_support_records` filter loop | Python per-candidate zip | **unroll** | Vectorize threshold (`counts >= ms·N`) with numpy and fuse `SupportRecord` construction; 17k-iteration Python loop at C-L3. Minor vs encoding. |
| Any kernel → GPU | flops/byte = 0.125 (bitmap), irregular (sparse) | **none** (GPU) | GPU verdict needs flops/byte: **0.125 ≪ 2** (the ≈2 ops/byte amortization bar from the README/port notes). Bitmap counting is pure `AND+popcount` streaming; PCIe/host-device transfer of 74–162 MB would dominate the 1–4 ms compute many times over. RTX 5090: irrelevant at 0.125 flops/byte. Sparse is random-access by construction — rejected for the same reason. |

---

## 9. Method notes (how every number was obtained)

- `pixi run bench` (flock `/tmp/mojo-bench.lock`) run verbatim; no concurrent benchmarks.
- Microbenchmarks: `/tmp/analyze_bench.py`, `/tmp/analyze_bench2.py` (plus inline
  one-shots); `time.perf_counter`, median of ≥11 (21/101 where noted), warmup ≥1
  (3/10). Upstream loaded via the same `importlib.util.spec_from_file_location` as
  `bench/bench.py`.
- `numpy` baselines coded in the same session (byte popcount table, `minimum.reduce`
  bool matrix); `plain Python` baseline is upstream apyori itself.
- Flops counted as scalar `AND`/`popcount`/comparison ops per word; bytes = `n·k·words·8`
  (+ output 8 B/candidate). Sparse bytes = pivot streaming + `log₂(pivot)` probes per
  candidate.
- Nothing in `src/` or `tests/` changed; this file (`bench/analysis.md`) is the sole addition.

