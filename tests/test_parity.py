from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import pathlib
from itertools import combinations, tee
from types import SimpleNamespace

import numpy as np
import pytest

import apyori
import apyori._lib as native_lib
from apyori._lib import count_bitmap
from apyori import (
    OrderedStatistic,
    RelationRecord,
    SupportRecord,
    TransactionManager,
)


def _load_upstream():
    distribution = importlib.metadata.distribution("apyori")
    source = pathlib.Path(distribution.locate_file("apyori.py"))
    spec = importlib.util.spec_from_file_location("upstream_apyori", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


upstream = _load_upstream()


def assert_records_equal(got, expected):
    assert len(got) == len(expected)
    for actual, reference in zip(got, expected):
        assert actual.items == reference.items
        assert actual.support == reference.support
        assert len(actual.ordered_statistics) == len(
            reference.ordered_statistics
        )
        for actual_stat, reference_stat in zip(
            actual.ordered_statistics, reference.ordered_statistics
        ):
            assert actual_stat.items_base == reference_stat.items_base
            assert actual_stat.items_add == reference_stat.items_add
            assert actual_stat.confidence == reference_stat.confidence
            assert actual_stat.lift == reference_stat.lift


def parity(transactions, **kwargs):
    ours, theirs = tee(transactions)
    got = list(apyori.apriori(ours, **kwargs))
    expected = list(upstream.apriori(theirs, **kwargs))
    assert_records_equal(got, expected)
    return got


def test_readme_market_basket_parity():
    transactions = [
        ["bread", "milk"],
        ["bread", "diaper", "beer", "eggs"],
        ["milk", "diaper", "beer", "cola"],
        ["bread", "milk", "diaper", "beer"],
        ["bread", "milk", "diaper", "cola"],
    ]
    records = parity(
        transactions,
        min_support=0.4,
        min_confidence=0.7,
        min_lift=1.0,
    )
    assert any(record.items == frozenset({"beer", "diaper"}) for record in records)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_support": 0.2},
        {"min_support": 0.15, "min_confidence": 0.6},
        {"min_support": 0.1, "min_lift": 1.1},
        {"min_support": 0.1, "max_length": 2},
        {"min_support": 0.1, "max_length": 1},
    ],
)
def test_random_integer_transactions_match_upstream(kwargs):
    rng = np.random.default_rng(7)
    transactions = [
        sorted(
            rng.choice(18, size=int(rng.integers(1, 8)), replace=False).tolist()
        )
        for _ in range(180)
    ]
    parity(transactions, **kwargs)


def test_duplicate_items_empty_transactions_and_generator_match():
    rows = [["b", "b", "a"], [], ["a"], ["c", "b"], [], ["a", "c", "c"]]
    parity((row for row in rows), min_support=0.25)


def test_empty_input_matches_upstream():
    assert parity([], min_support=0.1) == []


def test_transaction_manager_surface_matches_upstream():
    rows = [["c", "a"], ["b", "c"], ["c"], []]
    ours = TransactionManager(rows)
    theirs = upstream.TransactionManager(rows)
    assert ours.items == theirs.items == ["a", "b", "c"]
    assert ours.num_transaction == theirs.num_transaction == 4
    for items in [[], ["a"], ["c"], ["a", "c"], ["missing"]]:
        assert ours.calc_support(items) == theirs.calc_support(items)
    assert ours.initial_candidates() == theirs.initial_candidates()
    assert TransactionManager.create(ours) is ours
    ours.add_transaction(["a", "b"])
    theirs.add_transaction(["a", "b"])
    assert ours.calc_support(["a", "b"]) == theirs.calc_support(["a", "b"])


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_create_next_candidates_matches_upstream(length):
    previous = {
        frozenset(("a", "b", "c")[: max(1, length - 1)]),
        frozenset(("a", "b", "d")[: max(1, length - 1)]),
        frozenset(("a", "c", "d")[: max(1, length - 1)]),
        frozenset(("b", "c", "d")[: max(1, length - 1)]),
    }
    assert apyori.create_next_candidates(
        previous, length
    ) == upstream.create_next_candidates(previous, length)


def test_support_record_generator_matches_upstream():
    rows = [["a", "b"], ["a", "c"], ["a", "b", "c"], ["b"]]
    ours = list(
        apyori.gen_support_records(
            TransactionManager(rows), 0.25, max_length=2
        )
    )
    theirs = list(
        upstream.gen_support_records(
            upstream.TransactionManager(rows), 0.25, max_length=2
        )
    )
    assert ours == theirs


def test_support_record_generator_accepts_duck_typed_manager():
    class Manager:
        num_transaction = 10

        def initial_candidates(self):
            return [frozenset({"a"}), frozenset({"b"})]

        def calc_support(self, candidate):
            return {frozenset({"a"}): 0.7, frozenset({"b"}): 0.1}.get(
                candidate, 0.0
            )

    records = list(
        apyori.gen_support_records(
            Manager(),
            0.5,
            max_length=1,
            _create_next_candidates=lambda *_: [],
        )
    )
    assert records == [SupportRecord(frozenset({"a"}), 0.7)]


def test_apriori_private_extension_hooks_match_upstream_contract():
    manager = TransactionManager([])
    support = SupportRecord(frozenset({"a", "b"}), 0.5)
    statistic = OrderedStatistic(
        frozenset({"a"}), frozenset({"b"}), 0.75, 1.25
    )

    def supports(received_manager, received_support, **kwargs):
        assert received_manager is manager
        assert received_support == 0.2
        assert kwargs["max_length"] == 2
        yield support

    def statistics(received_manager, received_record):
        assert received_manager is manager
        assert received_record == support
        yield statistic

    def filter_stats(values, **kwargs):
        assert list(values) == [statistic]
        assert kwargs == {"min_confidence": 0.6, "min_lift": 1.1}
        yield statistic

    assert list(
        apyori.apriori(
            manager,
            min_support=0.2,
            min_confidence=0.6,
            min_lift=1.1,
            max_length=2,
            _gen_support_records=supports,
            _gen_ordered_statistics=statistics,
            _filter_ordered_statistics=filter_stats,
        )
    ) == [RelationRecord(support.items, support.support, [statistic])]


def test_ordered_statistics_and_filter_match_upstream():
    rows = [["a", "b"], ["a"], ["a", "b"], ["b"]]
    record = SupportRecord(frozenset({"a", "b"}), 0.5)
    ours = list(apyori.gen_ordered_statistics(TransactionManager(rows), record))
    theirs = list(
        upstream.gen_ordered_statistics(
            upstream.TransactionManager(rows),
            upstream.SupportRecord(record.items, record.support),
        )
    )
    assert ours == theirs
    assert list(
        apyori.filter_ordered_statistics(
            ours, min_confidence=0.7, min_lift=0.9
        )
    ) == list(
        upstream.filter_ordered_statistics(
            theirs, min_confidence=0.7, min_lift=0.9
        )
    )


@pytest.mark.parametrize("value", [0, -0.1, -10])
def test_invalid_min_support_matches_upstream(value):
    with pytest.raises(ValueError, match="minimum support must be > 0"):
        list(apyori.apriori([["a"]], min_support=value))


def test_namedtuple_shapes_match_upstream():
    assert SupportRecord._fields == upstream.SupportRecord._fields
    assert RelationRecord._fields == upstream.RelationRecord._fields
    assert OrderedStatistic._fields == upstream.OrderedStatistic._fields


def test_load_transactions_matches_upstream():
    text = "bread,milk\n\nbeer,diaper\n"
    assert list(apyori.load_transactions(io.StringIO(text), delimiter=",")) == list(
        upstream.load_transactions(io.StringIO(text), delimiter=",")
    )


def test_json_dump_matches_upstream():
    record = RelationRecord(
        frozenset({"bread", "milk"}),
        0.5,
        [
            OrderedStatistic(
                frozenset({"bread"}), frozenset({"milk"}), 0.75, 1.25
            )
        ],
    )
    ours = io.StringIO()
    theirs = io.StringIO()
    apyori.dump_as_json(record, ours)
    upstream.dump_as_json(record, theirs)
    assert json.loads(ours.getvalue()) == json.loads(theirs.getvalue())
    assert ours.getvalue() == theirs.getvalue()


def test_tsv_dump_matches_upstream():
    record = RelationRecord(
        frozenset({"bread", "milk"}),
        0.5,
        [
            OrderedStatistic(
                frozenset({"bread"}), frozenset({"milk"}), 0.75, 1.25
            ),
            OrderedStatistic(
                frozenset(), frozenset({"bread", "milk"}), 0.5, 1.0
            ),
        ],
    )
    ours = io.StringIO()
    theirs = io.StringIO()
    apyori.dump_as_two_item_tsv(record, ours)
    upstream.dump_as_two_item_tsv(record, theirs)
    assert ours.getvalue() == theirs.getvalue()


def test_parse_args_matches_upstream():
    arguments = [
        "--max-length",
        "3",
        "--min-support",
        "0.25",
        "--min-confidence",
        "0.6",
        "--min-lift",
        "1.1",
        "--delimiter",
        ",",
        "--out-format",
        "tsv",
    ]
    ours = apyori.parse_args(arguments)
    theirs = upstream.parse_args(arguments)
    for name in [
        "max_length",
        "min_support",
        "min_confidence",
        "min_lift",
        "delimiter",
        "out_format",
    ]:
        assert getattr(ours, name) == getattr(theirs, name)
    assert ours.output_func.__name__ == theirs.output_func.__name__


def test_main_extension_hooks_match_upstream_contract():
    destination = io.StringIO()
    args = SimpleNamespace(
        input=[io.StringIO("a,b\n"), io.StringIO("a,c\n")],
        output=destination,
        max_length=2,
        min_support=0.5,
        min_confidence=0.2,
        delimiter=",",
        output_func=lambda record, stream: stream.write(record),
    )

    def loader(lines, **kwargs):
        assert kwargs == {"delimiter": ","}
        for line in lines:
            yield line.strip().split(",")

    def miner(rows, **kwargs):
        assert list(rows) == [["a", "b"], ["a", "c"]]
        assert kwargs == {
            "max_length": 2,
            "min_support": 0.5,
            "min_confidence": 0.2,
        }
        yield "first"
        yield "second"

    apyori.main(
        _parse_args=lambda _: args,
        _load_transactions=loader,
        _apriori=miner,
    )
    assert destination.getvalue() == "firstsecond"


def test_sparse_kernel_path_matches_upstream(monkeypatch):
    monkeypatch.setattr(apyori, "_BITMAP_LIMIT_BYTES", 0)
    rows = [[item for item in range(40) if (item + row) % 7 == 0] for row in range(120)]
    parity(rows, min_support=0.02, max_length=3)


def test_bitmap_simd_tail():
    row_sets = [
        frozenset(
            item
            for item in range(24)
            if (row_index * 3 + item * 5) % 11 < 3
        )
        for row_index in range(1089)
    ]
    manager = TransactionManager(row_sets)
    for length, candidate_count in ((1, 255), (2, 255), (2, 256)):
        choices = [
            frozenset(candidate)
            for candidate in combinations(range(24), length)
        ]
        candidates = [
            choices[index % len(choices)]
            for index in range(candidate_count)
        ]
        expected = np.fromiter(
            (
                sum(candidate.issubset(row) for row in row_sets)
                for candidate in candidates
            ),
            dtype=np.int64,
            count=candidate_count,
        )
        np.testing.assert_array_equal(
            manager._count_candidates(candidates, length),
            expected,
        )


def test_singletons_skip_native_index_allocation():
    rows = [[0, 1], [0], [1, 2], [0, 2], []]
    manager = TransactionManager(rows)
    candidates = manager.initial_candidates()
    np.testing.assert_array_equal(
        manager._count_candidates(candidates, 1),
        np.array([3, 2, 2], dtype=np.int64),
    )
    assert manager._native_cache is None


def test_bitmap_parallel_threshold_keeps_small_work_serial(monkeypatch):
    candidates = np.array([0, 1, 0, 1], dtype=np.int64)
    bitmaps = np.full((2, 1), np.iinfo(np.uint64).max, dtype=np.uint64)
    counts = np.empty(2, dtype=np.int64)

    def unexpected_pool():
        raise AssertionError("small native count used the thread pool")

    monkeypatch.setattr(native_lib, "_PARALLEL_BITMAP_WORK", 5)
    monkeypatch.setattr(native_lib, "_PARALLEL_CANDIDATES", 2)
    monkeypatch.setattr(native_lib, "_pool", unexpected_pool)
    native_lib.count_bitmap(candidates, 2, 2, bitmaps, 1, counts)
    np.testing.assert_array_equal(counts, np.array([64, 64]))


def test_bitmap_parallel_threshold_splits_large_work(monkeypatch):
    candidates = np.array([0, 1, 0, 1], dtype=np.int64)
    bitmaps = np.full((2, 1), np.iinfo(np.uint64).max, dtype=np.uint64)
    counts = np.empty(2, dtype=np.int64)
    calls = []

    class Pool:
        def map(self, function, parts):
            parts = list(parts)
            calls.extend(parts)
            return map(function, parts)

    monkeypatch.setattr(native_lib, "_PARALLEL_BITMAP_WORK", 4)
    monkeypatch.setattr(native_lib, "_PARALLEL_CANDIDATES", 2)
    monkeypatch.setattr(native_lib, "_PARALLEL_WORKERS", 2)
    monkeypatch.setattr(native_lib, "_pool", Pool)
    native_lib.count_bitmap(candidates, 2, 2, bitmaps, 1, counts)
    np.testing.assert_array_equal(counts, np.array([64, 64]))
    assert calls == [(0, 1), (1, 2)]


def test_native_support_cache_is_invalidated_by_added_transaction():
    manager = TransactionManager([["a"], ["b"]])
    list(apyori.gen_support_records(manager, 0.1, max_length=1))
    item = frozenset({"a"})
    assert manager.calc_support(item) == 0.5
    manager.add_transaction(["a"])
    assert manager.calc_support(item) == 2 / 3


def test_ffi_rejects_wrong_dtype_shape_bounds_and_read_only_output():
    candidates = np.array([0, 1], dtype=np.int64)
    bitmaps = np.zeros((2, 1), dtype=np.uint64)
    counts = np.empty(1, dtype=np.int64)
    with pytest.raises(TypeError, match="dtype int64"):
        count_bitmap(candidates.astype(np.int32), 1, 2, bitmaps, 1, counts)
    with pytest.raises(ValueError, match="candidates size"):
        count_bitmap(candidates, 1, 1, bitmaps, 1, counts)
    with pytest.raises(ValueError, match="item ID"):
        count_bitmap(np.array([2], dtype=np.int64), 1, 1, bitmaps, 1, counts)
    counts.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        count_bitmap(candidates, 1, 2, bitmaps, 1, counts)
