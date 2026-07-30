"""A Mojo-accelerated, API-compatible port of apyori 1.1.2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import namedtuple
from itertools import chain, combinations

import numpy as np

from ._lib import build_bitmaps, count_bitmap, count_sparse

__version__ = "1.1.2"
__mojo_version__ = "0.1.0"
__author__ = "Yu Mochizuki; Mojo port by Lee Penkman"
__author_email__ = "ymoch.dev@gmail.com"

_BITMAP_LIMIT_BYTES = 256 * 1024 * 1024

SupportRecord = namedtuple("SupportRecord", ("items", "support"))
RelationRecord = namedtuple(
    "RelationRecord", SupportRecord._fields + ("ordered_statistics",)
)
OrderedStatistic = namedtuple(
    "OrderedStatistic", ("items_base", "items_add", "confidence", "lift")
)


class TransactionManager:
    """Store transactions and answer itemset support queries."""

    _mojo_native = True

    def __init__(self, transactions):
        self.__num_transaction = 0
        self.__items = []
        self.__transaction_index_map = {}
        self._native_cache = None
        self._support_cache = {}
        if type(self) is not TransactionManager:
            for transaction in transactions:
                self.add_transaction(transaction)
            return
        transaction_index_map = self.__transaction_index_map
        items = self.__items
        for transaction_index, transaction in enumerate(transactions):
            for item in transaction:
                indexes = transaction_index_map.get(item)
                if indexes is None:
                    items.append(item)
                    indexes = set()
                    transaction_index_map[item] = indexes
                indexes.add(transaction_index)
            self.__num_transaction = transaction_index + 1

    def add_transaction(self, transaction):
        for item in transaction:
            if item not in self.__transaction_index_map:
                self.__items.append(item)
                self.__transaction_index_map[item] = set()
            self.__transaction_index_map[item].add(self.__num_transaction)
        self.__num_transaction += 1
        self._native_cache = None
        self._support_cache.clear()

    def calc_support(self, items):
        if not items:
            return 1.0
        if not self.num_transaction:
            return 0.0
        if isinstance(items, frozenset):
            cached = self._support_cache.get(items)
            if cached is not None:
                return cached
        sum_indexes = None
        for item in items:
            indexes = self.__transaction_index_map.get(item)
            if indexes is None:
                return 0.0
            if sum_indexes is None:
                sum_indexes = indexes
            else:
                sum_indexes = sum_indexes.intersection(indexes)
        support = float(len(sum_indexes)) / self.__num_transaction
        if isinstance(items, frozenset):
            self._support_cache[items] = support
        return support

    def initial_candidates(self):
        return [frozenset([item]) for item in self.items]

    @property
    def num_transaction(self):
        return self.__num_transaction

    @property
    def items(self):
        return sorted(self.__items)

    @staticmethod
    def create(transactions):
        if isinstance(transactions, TransactionManager):
            return transactions
        return TransactionManager(transactions)

    def _native_index(self):
        if self._native_cache is not None:
            return self._native_cache
        sorted_items = self.items
        codes = {item: index for index, item in enumerate(sorted_items)}
        offsets = np.empty(len(sorted_items) + 1, dtype=np.int64)
        offsets[0] = 0
        size = 0
        for index, item in enumerate(sorted_items):
            size += len(self.__transaction_index_map[item])
            offsets[index + 1] = size
        words = (self.num_transaction + 63) // 64
        bitmap_bytes = len(sorted_items) * words * np.dtype(np.uint64).itemsize
        use_bitmaps = size and words and bitmap_bytes <= _BITMAP_LIMIT_BYTES
        memberships = (
            (self.__transaction_index_map[item] for item in sorted_items)
            if use_bitmaps
            else (
                sorted(self.__transaction_index_map[item])
                for item in sorted_items
            )
        )
        tids = np.fromiter(
            chain.from_iterable(memberships),
            dtype=np.int64,
            count=size,
        )
        bitmaps = None
        if use_bitmaps:
            bitmaps = np.zeros((len(sorted_items), words), dtype=np.uint64)
            build_bitmaps(tids, offsets, len(sorted_items), words, bitmaps)
        self._native_cache = codes, tids, offsets, bitmaps, words
        return self._native_cache

    def _count_candidates(self, candidates, length):
        if not candidates:
            return np.empty(0, dtype=np.int64)
        codes, tids, offsets, bitmaps, words = self._native_index()
        encoded = np.fromiter(
            (codes[item] for candidate in candidates for item in candidate),
            dtype=np.int64,
            count=len(candidates) * length,
        )
        counts = np.empty(len(candidates), dtype=np.int64)
        if bitmaps is not None:
            count_bitmap(
                encoded,
                len(candidates),
                length,
                bitmaps,
                words,
                counts,
            )
        else:
            count_sparse(
                encoded,
                len(candidates),
                length,
                tids,
                offsets,
                counts,
            )
        return counts


def create_next_candidates(prev_candidates, length):
    items = sorted(frozenset(chain.from_iterable(prev_candidates)))
    temporary = (frozenset(value) for value in combinations(items, length))
    if length < 3:
        return list(temporary)
    return [
        candidate
        for candidate in temporary
        if all(
            frozenset(value) in prev_candidates
            for value in combinations(candidate, length - 1)
        )
    ]


def gen_support_records(transaction_manager, min_support, **kwargs):
    max_length = kwargs.get("max_length")
    next_candidates = kwargs.get(
        "_create_next_candidates", create_next_candidates
    )
    candidates = transaction_manager.initial_candidates()
    length = 1
    while candidates:
        relations = set()
        if getattr(transaction_manager, "_mojo_native", False) is True:
            counts = transaction_manager._count_candidates(candidates, length)
            denominator = transaction_manager.num_transaction
            supports = (
                float(count) / denominator if denominator else 0.0
                for count in counts
            )
        else:
            supports = (
                transaction_manager.calc_support(candidate)
                for candidate in candidates
            )
        for relation_candidate, support in zip(candidates, supports):
            if support < min_support:
                continue
            candidate_set = frozenset(relation_candidate)
            if getattr(transaction_manager, "_mojo_native", False) is True:
                transaction_manager._support_cache[candidate_set] = support
            relations.add(candidate_set)
            yield SupportRecord(candidate_set, support)
        length += 1
        if max_length and length > max_length:
            break
        candidates = next_candidates(relations, length)


def gen_ordered_statistics(transaction_manager, record):
    items = record.items
    sorted_items = sorted(items)
    for base_length in range(len(items)):
        for combination_set in combinations(sorted_items, base_length):
            items_base = frozenset(combination_set)
            items_add = frozenset(items.difference(items_base))
            confidence = record.support / transaction_manager.calc_support(
                items_base
            )
            lift = confidence / transaction_manager.calc_support(items_add)
            yield OrderedStatistic(
                frozenset(items_base),
                frozenset(items_add),
                confidence,
                lift,
            )


def filter_ordered_statistics(ordered_statistics, **kwargs):
    min_confidence = kwargs.get("min_confidence", 0.0)
    min_lift = kwargs.get("min_lift", 0.0)
    for ordered_statistic in ordered_statistics:
        if ordered_statistic.confidence < min_confidence:
            continue
        if ordered_statistic.lift < min_lift:
            continue
        yield ordered_statistic


def apriori(transactions, **kwargs):
    min_support = kwargs.get("min_support", 0.1)
    min_confidence = kwargs.get("min_confidence", 0.0)
    min_lift = kwargs.get("min_lift", 0.0)
    max_length = kwargs.get("max_length", None)
    if min_support <= 0:
        raise ValueError("minimum support must be > 0")

    support_generator = kwargs.get("_gen_support_records", gen_support_records)
    statistic_generator = kwargs.get(
        "_gen_ordered_statistics", gen_ordered_statistics
    )
    statistic_filter = kwargs.get(
        "_filter_ordered_statistics", filter_ordered_statistics
    )
    transaction_manager = TransactionManager.create(transactions)
    support_records = support_generator(
        transaction_manager, min_support, max_length=max_length
    )
    for support_record in support_records:
        ordered_statistics = list(
            statistic_filter(
                statistic_generator(transaction_manager, support_record),
                min_confidence=min_confidence,
                min_lift=min_lift,
            )
        )
        if not ordered_statistics:
            continue
        yield RelationRecord(
            support_record.items,
            support_record.support,
            ordered_statistics,
        )


def parse_args(argv):
    output_funcs = {"json": dump_as_json, "tsv": dump_as_two_item_tsv}
    default_output_func_key = "json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {0}".format(__version__),
    )
    parser.add_argument(
        "input",
        metavar="inpath",
        nargs="*",
        help="Input transaction file (default: stdin).",
        type=argparse.FileType("r"),
        default=[sys.stdin],
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="outpath",
        help="Output file (default: stdout).",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )
    parser.add_argument(
        "-l",
        "--max-length",
        metavar="int",
        help="Max length of relations (default: infinite).",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--min-support",
        metavar="float",
        help="Minimum support ratio (must be > 0, default: 0.1).",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        metavar="float",
        help="Minimum confidence (default: 0.5).",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-t",
        "--min-lift",
        metavar="float",
        help="Minimum lift (default: 0.0).",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        metavar="str",
        help="Delimiter for items of transactions (default: tab).",
        type=str,
        default="\t",
    )
    parser.add_argument(
        "-f",
        "--out-format",
        metavar="str",
        help="Output format ({0}; default: {1}).".format(
            ", ".join(output_funcs.keys()), default_output_func_key
        ),
        type=str,
        choices=output_funcs.keys(),
        default=default_output_func_key,
    )
    args = parser.parse_args(argv)
    args.output_func = output_funcs[args.out_format]
    return args


def load_transactions(input_file, **kwargs):
    delimiter = kwargs.get("delimiter", "\t")
    for transaction in csv.reader(input_file, delimiter=delimiter):
        yield transaction if transaction else [""]


def dump_as_json(record, output_file):
    def default_func(value):
        if isinstance(value, frozenset):
            return sorted(value)
        raise TypeError(repr(value) + " is not JSON serializable")

    converted_record = record._replace(
        ordered_statistics=[
            value._asdict() for value in record.ordered_statistics
        ]
    )
    json.dump(
        converted_record._asdict(),
        output_file,
        default=default_func,
        ensure_ascii=False,
    )
    output_file.write(os.linesep)


def dump_as_two_item_tsv(record, output_file):
    for ordered_stats in record.ordered_statistics:
        if len(ordered_stats.items_base) != 1:
            continue
        if len(ordered_stats.items_add) != 1:
            continue
        output_file.write(
            "{0}\t{1}\t{2:.8f}\t{3:.8f}\t{4:.8f}{5}".format(
                list(ordered_stats.items_base)[0],
                list(ordered_stats.items_add)[0],
                record.support,
                ordered_stats.confidence,
                ordered_stats.lift,
                os.linesep,
            )
        )


def main(**kwargs):
    argument_parser = kwargs.get("_parse_args", parse_args)
    transaction_loader = kwargs.get("_load_transactions", load_transactions)
    apriori_function = kwargs.get("_apriori", apriori)
    args = argument_parser(sys.argv[1:])
    transactions = transaction_loader(
        chain(*args.input), delimiter=args.delimiter
    )
    result = apriori_function(
        transactions,
        max_length=args.max_length,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )
    for record in result:
        args.output_func(record, args.output)


__all__ = [
    "OrderedStatistic",
    "RelationRecord",
    "SupportRecord",
    "TransactionManager",
    "apriori",
    "create_next_candidates",
    "dump_as_json",
    "dump_as_two_item_tsv",
    "filter_ordered_statistics",
    "gen_ordered_statistics",
    "gen_support_records",
    "load_transactions",
    "main",
    "parse_args",
]
