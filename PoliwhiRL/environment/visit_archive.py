# -*- coding: utf-8 -*-
"""Persistent cell-visit archive driving Phase-4 frontier novelty rewards.

The archive maps each *quantised cell* ``(map_bank, map_num, x // CELL, y // CELL)``
to the cumulative number of times the policy has visited it. The reward
calculator pays a novelty bonus that decays with visit count, so the
gradient naturally points the policy at under-explored cells without
needing a hand-authored ``terminate_on`` predicate.

There is no per-episode "target cell" sampler — the frontier is implicit
in the reward landscape: fresh cells pay full bonus, oft-visited cells
pay near-zero. Cells stop being attractive on their own as visits
accumulate, and the policy moves on.

Lifetime: an archive instance is owned by the :class:`PyBoyEnvironment`
and lives for the lifetime of one training process. ``Rewards`` receives
a reference at construction and reads / writes through it. Because
``env.reset()`` re-instantiates ``Rewards``, the env explicitly carries
the archive forward across resets so visit counts accumulate across
episodes (not just within an episode).

Quantisation: ``CELL_SIZE = 2`` tiles. Coarser than per-tile, finer than
per-map. Keeps the archive bounded and prevents the policy from farming
novelty by wiggling on a 1-tile boundary, while staying granular enough
to credit movement within a single sub-region. The earlier value (4) was
too coarse for indoor stages where 2-tile shifts represent meaningful
progress.
"""

from collections import defaultdict

CELL_SIZE = 2


def quantise(map_bank, map_num, x, y):
    """Map a raw player position to its archive key. Quantising x/y by
    ``CELL_SIZE`` keeps the archive bounded and is fine-grained enough
    to distinguish meaningful sub-map regions."""
    return (int(map_bank), int(map_num), int(x) // CELL_SIZE, int(y) // CELL_SIZE)


class VisitArchive:
    def __init__(self):
        # cell -> int. defaultdict means count() returns 0 for unseen
        # cells without needing membership checks at every call site.
        #
        # NOTE: as of the episodic-novelty rework, per-cell counts are no
        # longer written by the reward calculator (frontier novelty is now
        # per-episode and stationary). ``_counts`` is retained for any
        # diagnostic / tooling caller but is not part of the training reward
        # path. The live training signal that uses this archive is the
        # *map-level* ledger below.
        self._counts = defaultdict(int)
        # (map_bank, map_num) -> number of times that map has been entered
        # fresh during the *training* portion across the whole run. Backs the
        # smoothly-decaying new_map first-discovery bonus so map-bouncing
        # stops paying after a handful of entries while genuine frontier maps
        # still pay on first discovery.
        self._map_counts = defaultdict(int)

    def record(self, map_bank, map_num, x, y):
        """Increment the visit count for the cell containing (x, y) on
        (map_bank, map_num). Returns the *new* count after recording."""
        key = quantise(map_bank, map_num, x, y)
        self._counts[key] += 1
        return self._counts[key]

    def record_map(self, map_bank, map_num):
        """Increment the run-wide entry count for (map_bank, map_num).
        Returns the new count. Written only by genuine training-segment map
        entries (never by action replay — see ``Rewards._replaying``)."""
        key = (int(map_bank), int(map_num))
        self._map_counts[key] += 1
        return self._map_counts[key]

    def map_count(self, map_bank, map_num):
        """Run-wide training entry count for (map_bank, map_num); 0 if never
        discovered."""
        return self._map_counts[(int(map_bank), int(map_num))]

    def count(self, map_bank, map_num, x, y):
        return self._counts[quantise(map_bank, map_num, x, y)]

    def cell_key(self, map_bank, map_num, x, y):
        return quantise(map_bank, map_num, x, y)

    def n_cells_seen(self):
        return len(self._counts)

    def total_visits(self):
        return sum(self._counts.values())

    def snapshot(self):
        """Plain-dict snapshot of the archive — JSON-serialisable."""
        return {
            f"{a}_{b}_{c}_{d}": v for (a, b, c, d), v in self._counts.items()
        }
