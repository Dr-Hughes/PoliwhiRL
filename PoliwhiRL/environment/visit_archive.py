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

Ownership: the AGENT owns the canonical archive. Each worker env holds a
read-only replica that the agent refreshes via ``load_state`` broadcasts
(once per rollout). ``Rewards`` never writes counts directly — it
accumulates the episode's genuinely-visited cell/map keys in pending sets
(``_cells_to_record`` / ``_maps_to_record``) which the worker reports in
``terminal_info`` at episode end; the agent merges them with
``merge_visits`` (one increment per cell/map per episode). This makes the
novelty landscape global across all workers AND persistent across
curriculum stages (the agent checkpoints the archive in ``info.pth``), so
the gradient always points at the run's true frontier rather than at 16
private ones. ``Rewards`` receives the env's replica at construction and
reads through it; replica staleness is bounded by one rollout and is
harmless (the once-per-episode gate handles within-episode farming).

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
        # Cell counts back the depleting frontier-novelty bonus; one
        # increment per cell per EPISODE (merged by the agent from the
        # per-episode pending sets, never per step).
        self._counts = defaultdict(int)
        # (map_bank, map_num) -> number of episodes that entered that map
        # fresh during the *training* portion across the whole run. Backs the
        # smoothly-decaying new_map first-discovery bonus so map-bouncing
        # stops paying after a handful of entries while genuine frontier maps
        # still pay on first discovery.
        self._map_counts = defaultdict(int)

    def merge_visits(self, cells, maps):
        """Merge ONE episode's genuinely-visited cell/map keys into the
        canonical counts (+1 each). Called by the agent with the pending
        sets the worker reported in terminal_info."""
        for key in cells or []:
            a, b, c, d = key
            self._counts[(int(a), int(b), int(c), int(d))] += 1
        for key in maps or []:
            a, b = key
            self._map_counts[(int(a), int(b))] += 1

    def to_state(self):
        """Serializable full-table state (pipes / torch.save)."""
        return {"cells": dict(self._counts), "maps": dict(self._map_counts)}

    def load_state(self, state):
        """Replace both tables with a broadcast/checkpointed state."""
        self._counts = defaultdict(int)
        self._counts.update(
            {tuple(k): int(v) for k, v in (state.get("cells") or {}).items()}
        )
        self._map_counts = defaultdict(int)
        self._map_counts.update(
            {tuple(k): int(v) for k, v in (state.get("maps") or {}).items()}
        )

    def map_count(self, map_bank, map_num):
        """Run-wide training entry count for (map_bank, map_num); 0 if never
        discovered. Non-mutating read (a defaultdict[] would insert zero
        entries and inflate n_cells_seen / to_state)."""
        return self._map_counts.get((int(map_bank), int(map_num)), 0)

    def count(self, map_bank, map_num, x, y):
        """Non-mutating read; 0 for never-visited cells."""
        return self._counts.get(quantise(map_bank, map_num, x, y), 0)

    def cell_key(self, map_bank, map_num, x, y):
        return quantise(map_bank, map_num, x, y)

    def decay(self, factor):
        """Multiply all cell visit counts by ``factor`` and drop counts that
        round down to zero.

        Called periodically by the agent when the archive growth rate
        approaches zero so previously saturated cells gradually become
        attractive again. Only cell counts are decayed — map discovery
        counts are a coarse ledger and should not be re-paid.

        ``factor`` must be in (0, 1). A value of 0.97 every 100 rollouts
        gives a half-life of ~2200 rollouts, enough to re-open heavily
        visited territory over a long curriculum without aggressively
        resetting the novelty landscape.
        """
        if not (0 < factor < 1):
            return
        for key in list(self._counts.keys()):
            new_val = int(self._counts[key] * factor)
            if new_val <= 0:
                del self._counts[key]
            else:
                self._counts[key] = new_val

    def n_cells_seen(self):
        return len(self._counts)

    def total_visits(self):
        return sum(self._counts.values())

    def snapshot(self):
        """Plain-dict snapshot of the archive — JSON-serialisable."""
        return {
            f"{a}_{b}_{c}_{d}": v for (a, b, c, d), v in self._counts.items()
        }
