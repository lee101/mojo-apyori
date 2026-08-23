"""Batched support-counting kernels for Apriori itemset mining."""

from std.bit import pop_count
from std.sys.info import simd_width_of

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.uint64]()


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def count_bitmap_range(
    candidates: I64Ptr,
    candidate_length: Int,
    bitmaps: U64Ptr,
    words: Int,
    counts: I64Ptr,
    start: Int,
    stop: Int,
):
    if candidate_length == 1:
        for candidate_index in range(start, stop):
            var item = Int(candidates[unsafe_offset=candidate_index])
            var total = 0
            var word = 0
            while word + W <= words:
                total += Int(
                    pop_count(
                        bitmaps.unsafe_load[width=W](item * words + word)
                    ).reduce_add()
                )
                word += W
            while word < words:
                total += Int(
                    pop_count(bitmaps[unsafe_offset=item * words + word])
                )
                word += 1
            counts[unsafe_offset=candidate_index] = Int64(total)
        return

    if candidate_length == 2:
        for candidate_index in range(start, stop):
            var candidate_offset = candidate_index * 2
            var first_offset = (
                Int(candidates[unsafe_offset=candidate_offset]) * words
            )
            var second_offset = (
                Int(candidates[unsafe_offset=candidate_offset + 1]) * words
            )
            var total = 0
            var word = 0
            while word + W <= words:
                var intersection = bitmaps.unsafe_load[width=W](
                    first_offset + word
                ) & bitmaps.unsafe_load[width=W](second_offset + word)
                total += Int(pop_count(intersection).reduce_add())
                word += W
            while word < words:
                total += Int(
                    pop_count(
                        bitmaps[unsafe_offset=first_offset + word]
                        & bitmaps[unsafe_offset=second_offset + word]
                    )
                )
                word += 1
            counts[unsafe_offset=candidate_index] = Int64(total)
        return

    for candidate_index in range(start, stop):
        var candidate_offset = candidate_index * candidate_length
        var first_item = Int(candidates[unsafe_offset=candidate_offset])
        var total = 0
        var word = 0
        while word + W <= words:
            var intersection = bitmaps.unsafe_load[width=W](
                first_item * words + word
            )
            for item_index in range(1, candidate_length):
                var item = Int(
                    candidates[unsafe_offset=candidate_offset + item_index]
                )
                intersection &= bitmaps.unsafe_load[width=W](
                    item * words + word
                )
            total += Int(pop_count(intersection).reduce_add())
            word += W
        while word < words:
            var intersection = bitmaps[unsafe_offset=first_item * words + word]
            for item_index in range(1, candidate_length):
                var item = Int(
                    candidates[unsafe_offset=candidate_offset + item_index]
                )
                intersection &= bitmaps[unsafe_offset=item * words + word]
            total += Int(pop_count(intersection))
            word += 1
        counts[unsafe_offset=candidate_index] = Int64(total)


def contains_tid(tids: I64Ptr, start: Int, stop: Int, target: Int64) -> Bool:
    var lo = start
    var hi = stop
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if tids[unsafe_offset=mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo < stop and tids[unsafe_offset=lo] == target


def count_sparse_range(
    candidates: I64Ptr,
    candidate_length: Int,
    tids: I64Ptr,
    offsets: I64Ptr,
    counts: I64Ptr,
    start: Int,
    stop: Int,
):
    for candidate_index in range(start, stop):
        var candidate_offset = candidate_index * candidate_length
        var pivot_position = 0
        var pivot_item = Int(candidates[unsafe_offset=candidate_offset])
        var pivot_size = Int(
            offsets[unsafe_offset=pivot_item + 1]
            - offsets[unsafe_offset=pivot_item]
        )
        for position in range(1, candidate_length):
            var item = Int(
                candidates[unsafe_offset=candidate_offset + position]
            )
            var item_size = Int(
                offsets[unsafe_offset=item + 1] - offsets[unsafe_offset=item]
            )
            if item_size < pivot_size:
                pivot_position = position
                pivot_item = item
                pivot_size = item_size

        var total = 0
        var pivot_start = Int(offsets[unsafe_offset=pivot_item])
        var pivot_stop = Int(offsets[unsafe_offset=pivot_item + 1])
        for tid_index in range(pivot_start, pivot_stop):
            var tid = tids[unsafe_offset=tid_index]
            var present = True
            for position in range(candidate_length):
                if position == pivot_position:
                    continue
                var item = Int(
                    candidates[unsafe_offset=candidate_offset + position]
                )
                if not contains_tid(
                    tids,
                    Int(offsets[unsafe_offset=item]),
                    Int(offsets[unsafe_offset=item + 1]),
                    tid,
                ):
                    present = False
                    break
            if present:
                total += 1
        counts[unsafe_offset=candidate_index] = Int64(total)


@export("map_build_bitmaps")
def map_build_bitmaps(
    tids_addr: Int,
    offsets_addr: Int,
    item_count: Int,
    words: Int,
    bitmaps_addr: Int,
) abi("C"):
    var tids = i64p(tids_addr)
    var offsets = i64p(offsets_addr)
    var bitmaps = u64p(bitmaps_addr)
    for item in range(item_count):
        for position in range(
            Int(offsets[unsafe_offset=item]),
            Int(offsets[unsafe_offset=item + 1]),
        ):
            var transaction = Int(tids[unsafe_offset=position])
            var word = transaction // 64
            var bit = transaction % 64
            bitmaps[unsafe_offset=item * words + word] |= UInt64(1) << UInt64(
                bit
            )


@export("map_count_bitmap")
def map_count_bitmap(
    candidates_addr: Int,
    candidate_count: Int,
    candidate_length: Int,
    bitmaps_addr: Int,
    words: Int,
    counts_addr: Int,
) abi("C"):
    var candidates = i64p(candidates_addr)
    var bitmaps = u64p(bitmaps_addr)
    var counts = i64p(counts_addr)
    count_bitmap_range(
        candidates,
        candidate_length,
        bitmaps,
        words,
        counts,
        0,
        candidate_count,
    )


@export("map_count_sparse")
def map_count_sparse(
    candidates_addr: Int,
    candidate_count: Int,
    candidate_length: Int,
    tids_addr: Int,
    offsets_addr: Int,
    counts_addr: Int,
) abi("C"):
    var candidates = i64p(candidates_addr)
    var tids = i64p(tids_addr)
    var offsets = i64p(offsets_addr)
    var counts = i64p(counts_addr)
    count_sparse_range(
        candidates,
        candidate_length,
        tids,
        offsets,
        counts,
        0,
        candidate_count,
    )
