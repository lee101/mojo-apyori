# VERIFY — independent reproduction of README benchmark figures (this session)

Date: 2026-11-09 session. Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz
(72 logical CPUs, from `/proc/cpuinfo`; same model string the README states).
Toolchain: pixi 0.76.1, mojo 1.2.0.dev2026092605, upstream apyori 1.1.2
(`importlib.metadata.version("apyori") == "1.1.2"`).
Load during session: 57.44 / 48.65 / 36.68 (shared box) — absolute ms vary with it.

## Commands run, in order (this session, fresh shell)

```
pixi run build && pixi run test   # exit 0; build ok, 34 passed in 0.86s
pixi run bench                    # Pixi task: flock /tmp/mojo-bench.lock python bench/bench.py (best-of-3 per workload), run 1
pixi run bench                    # run 2
# independent rerun (not the bench script): /tmp/verify_independent.py, best-of-5 minima, exact bench generators/kwargs
pixi run python /tmp/verify_independent.py
```

## What reproduced

Two flock-serialized `pixi run bench` runs and one independent best-of-5 rerun
all agreed with the README on all four rows, all within 20% of the README figure:

| workload | README | this session bench run 1 | this session bench run 2 | independent best-of-5 |
| --- | ---: | ---: | ---: | ---: |
| 250 baskets, 24 items, width 4, max length 2 | 2.3 ms / 2.6 ms / 1.13x | 2.4 ms / 2.7 ms / 1.14x | 2.5 ms / 2.8 ms / 1.15x | 2.2 ms / 2.6 ms / 1.17x |
| 10k baskets, 8 items, width 2, max length 1 | 3.7 ms / 4.3 ms / 1.16x | 3.7 ms / 4.4 ms / 1.17x | 3.8 ms / 4.5 ms / 1.19x | 4.2 ms / 4.5 ms / 1.06x |
| 25k baskets, 48 items, width 10, max length 3 | 160.8 ms / 4075.4 ms / 25.34x | 153.6 ms / 3984.1 ms / 25.93x | 155.6 ms / 4085.1 ms / 26.26x | 155.7 ms / 4157.7 ms / 26.71x |
| 60k baskets, 100 items, width 5, max length 2 | 169.9 ms / 998.4 ms / 5.88x | 165.3 ms / 1029.3 ms / 6.23x | 162.0 ms / 1082.0 ms / 6.68x | 148.1 ms / 1028.4 ms / 6.95x |

Columns are mojo-apyori / upstream apyori / speedup (upstream/mojo).
Max deviation from the README figure across both bench runs: 13.6% (60k speedup,
run 2). Per-row measured deviations (run 1): 0.0–6.0%; (run 2): 2.6–13.6%.
Every README figure is within 20% of both measured runs, so per the verify rules
no row is corrected.

Notes on the independent rerun:

- It builds workloads with the exact generators the bench uses
  (`transactions(0/3/1/2, ...)` via `np.random.default_rng`), times mojo first
  then upstream (best-of-5), and asserts `mojo_result == python_result`.
- It agrees on all four rows within 20% as well; the small-row ratios
  (1.06x on the 10k workload vs 1.17–1.19x in the bench runs) confirm the
  small-row margin (~1.1x) is real but small and load-sensitive, while the
  large-row speedups (25–27x, 6–7x) are robust and reproduce in every run.

## Same-input / no-skipped-work audit

- `bench/bench.py` builds each workload once
  (`transactions(seed, rows, items, width)`), passes the same `rows` list and the
  same `kwargs` dict to both `apyori.apriori` and `upstream.apriori`, and asserts
  `mojo_result == python_result` before printing a ratio. No per-side input
  difference, no skipped levels, no weak parity check that would mask skipping.
- Independently re-checked this session: for all four workloads the mojo and
  upstream outputs are exactly equal (200, 8, 1176, 5050 rules respectively, with
  552 / 8 / 3432 / 14950 ordered statistics). A brute-force recount of 5 random
  rules per workload (subset test against every transaction) matched the reported
  supports to < 1e-12 (`ALL OK`), so neither side is skipping counting work.
- The two modules are distinct files (`python/apyori/__init__.py` vs
  `.pixi/envs/default/lib/python3.13/site-packages/apyori.py`);
  `bench.py` loads upstream from its installed file location via
  `spec_from_file_location`, so they coexist under different module objects.
  The upstream `apriori`, `gen_support_records`, and `calc_support` are the
  stock apyori 1.1.2 implementations (no patching, no `max_length` tricks).

## Slower rows

There are none in the reproduced numbers, and none in the README. No row was
removed or reworded. If a future run shows the port slower on a row, it stays
slower in the table per the verify rules.

## Tests still assert parity with upstream

`git log -p -- tests/` shows only two commits touching tests: the initial port
(`a8a4b66`) and the accel pass (`f8dc387`). The accel pass only **added**
assertions (singleton-skip, parallel-threshold serial/parallel split,
cache-invalidation) and renamed one test
(`test_bitmap_simd_tail_at_serial_parallel_boundary` →
`test_bitmap_simd_tail`) with no assertion change. No parity assertion was
removed, no tolerance was loosened. Current `tests/test_parity.py` uses exact
equality for parity: `actual.support == reference.support`,
`actual.confidence == reference.confidence`, `np.testing.assert_array_equal`,
`assert ours == theirs`. No `pytest.approx`, `isclose`, `rel=`, or tolerance
anywhere in the file (grep-verified). All 34 tests pass.

Nothing needed restoring.

## What changed in this session

- **README.md**: no change. Every figure was reproduced within 20% by the
  flock-serialized `pixi run bench` protocol (two independent runs) and by an
  independent best-of-5 rerun, so there was nothing to correct, and inventing
  figures is prohibited.
- **Code**: no change. The verify rules allow code changes only to fix a bug
  shown by a failing test; `pixi run build && pixi run test` was green before
  and after, and no bug was found.
- **bench/verify.md**: this file.

## What could not be fully reproduced (and why it doesn't change the table)

1. The small-row speedups (~1.13–1.19x) sit within shared-machine noise: the
   independent rerun gives 1.06x on the 10k workload and the bench runs give
   1.13–1.19x. The direction (port faster) and the ~1.1x magnitude reproduce in
   every run this session, but the margin is small relative to run-to-run
   variance on a loaded box (load ~57 on 72 cpus). This does not justify a
   correction under the 20% rule; it is disclosed here so readers do not treat
   the small rows as decisive.
2. Absolute milliseconds vary ±20% run to run under load (e.g. 25k upstream
   3984–4158 ms; 60k speedup 6.23–6.95x). The README numbers sit inside that band.

## Final state

`pixi run build` — exit 0. `pixi run test` — 34 passed.
