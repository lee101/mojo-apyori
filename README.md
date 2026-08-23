# mojo-apyori

`mojo-apyori` is a Mojo-accelerated port of
[apyori](https://pypi.org/project/apyori/) for Apriori frequent-itemset mining
and association-rule generation. It keeps the upstream `apyori` import,
function names, signatures, namedtuple results, thresholds, ordering, and CLI
helpers while moving batched support counting into compiled Mojo.

## Scope and upstream coverage

This port covers the complete public API shipped in upstream apyori 1.1.2:

- `apriori`
- `TransactionManager`
- `create_next_candidates` and `gen_support_records`
- `gen_ordered_statistics` and `filter_ordered_statistics`
- `SupportRecord`, `RelationRecord`, and `OrderedStatistic`
- transaction loading, JSON/TSV output, argument parsing, and `main`

Parity tests exercise every function listed above against upstream, including
edge cases, extension hooks, both native kernels, SIMD tails, and the serial to
parallel boundary. The implementation accepts the same arbitrary hashable,
mutually sortable Python items as upstream.

This is API compatibility, not packaging compatibility: the project currently
builds and runs from a Linux source checkout with Pixi and does not publish
binary wheels. It also does not add closed or maximal itemsets, FP-growth,
streaming updates, weighted transactions, distributed mining, or GPU support.
Those features are outside upstream apyori's API and this repository's scope.

## Install

```bash
git clone https://github.com/lee101/mojo-apyori.git
cd mojo-apyori
pixi install
pixi run build
pixi run test
```

The build emits `dist/libmojo-apyori.so`. Set `MOJO_APYORI_LIB` to use a shared
library at another location.

## Usage

```python
from apyori import apriori

transactions = [
    ["bread", "milk"],
    ["bread", "diaper", "beer", "eggs"],
    ["milk", "diaper", "beer", "cola"],
    ["bread", "milk", "diaper", "beer"],
    ["bread", "milk", "diaper", "cola"],
]

rules = list(
    apriori(
        transactions,
        min_support=0.4,
        min_confidence=0.7,
        min_lift=1.0,
    )
)
for rule in rules:
    print(rule.items, rule.support)
```

## Benchmarks

Run the serialized, machine-wide-flocked benchmark with:

```bash
pixi run bench
```

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz (72 logical CPUs):

| workload | mojo-apyori | upstream apyori 1.1.2 | result |
| --- | ---: | ---: | ---: |
| 250 baskets, 24 items, width 4, max length 2 | 2.3 ms | 2.6 ms | 1.13x faster |
| 10k baskets, 8 items, width 2, max length 1 | 3.7 ms | 4.3 ms | 1.16x faster |
| 25k baskets, 48 items, width 10, max length 3 | 160.8 ms | 4075.4 ms | 25.34x faster |
| 60k baskets, 100 items, width 5, max length 2 | 169.9 ms | 998.4 ms | 5.88x faster |

Singleton support uses membership cardinalities directly and does not build a
native index. This removes the allocation overhead that previously made the
10k-row singleton workload slower than upstream. The larger three-item workload
is dominated by SIMD bitmap intersection and wins decisively. Python-side
transaction ingestion and result construction account for most of the smaller
workloads.

## How it works

Python retains ownership of all objects and produces the same `frozenset` and
namedtuple results as apyori. For each candidate level it maps the sorted
Python items to contiguous `int64` IDs and makes one native counting call.
Before that call, the binding validates dtypes, dimensions, bounds, alignment,
contiguity, nullability, and output mutability; local references keep every
NumPy allocation alive until Mojo returns. Buffers cross the C ABI as integer
addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside non-parametric
`@export(...) ... abi("C")` wrappers.

The native index is vertical: each item owns an `int64` transaction-ID segment.
When an item-by-transaction bitmap needs no more than 256 MiB, Mojo materializes
contiguous `uint64` rows and counts candidates with SIMD AND and
population-count operations. Large independent candidate batches are divided
into zero-copy contiguous views and counted by up to eight concurrent native
calls; explicit work thresholds keep smaller bitmap and sparse batches serial.
Larger sparse problems keep sorted segments in the compact vertical layout and
use a shortest-list, binary-intersection kernel instead. All arrays and native
output storage are allocated and owned by NumPy; Mojo allocates nothing across
the FFI boundary.
Supports produced by native counting are reused during association-rule
generation instead of recomputing the same itemset intersections in Python.

GPU acceleration is intentionally not included. Bitmap counting performs only
an AND and population count while reading at least two 64-bit rows, far below
the roughly 2 operations per byte needed to amortize device transfers. The
sparse kernel is likewise dominated by irregular binary-search memory access,
so neither kernel is a suitable GPU target.

## License

MIT. The compatibility API derives from Yu Mochizuki's MIT-licensed apyori
1.1.2.
