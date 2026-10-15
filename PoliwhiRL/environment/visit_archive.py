# -*- coding: utf-8 -*-
"""Persistent cell-visit archive driving Phase-4 frontier novelty rewards.

The archive maps each *quantised cell* ``(map_bank, map_num, x // CELL, y // CELL)``
to completed-episode reach counts, tracked independently for honest and
seeded episodes as well as in aggregate. The reward calculator can decay
novelty against the current episode's provenance without seeded starts
depleting canonical-start exploration.

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

from PoliwhiRL.checkpoints import is_recordable_checkpoint

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
        # Completed-episode cell reach counts split by episode provenance.
        # ``_counts`` remains the aggregate compatibility view used by archive
        # size/Go-Explore callers and old checkpoints.
        self._honest_counts = defaultdict(int)
        self._seeded_counts = defaultdict(int)
        # (map_bank, map_num) -> number of episodes that entered that map
        # fresh during the *training* portion across the whole run. Backs the
        # smoothly-decaying new_map first-discovery bonus so map-bouncing
        # stops paying after a handful of entries while genuine frontier maps
        # still pay on first discovery.
        self._map_counts = defaultdict(int)
        self._honest_map_counts = defaultdict(int)
        self._seeded_map_counts = defaultdict(int)

        # Run-wide (training-time) "has any completed episode so far ever
        # achieved this" ledger — backs the discovery log (rewards.py's
        # _discoveries_this_episode) for milestone types that have no
        # persistent in-game state to key off. Every episode restarts from
        # the same save-state (no snapshot seeding), so flags/pokedex/level/
        # key-items are NOT persistent game-world facts the way map/cell
        # visits are — they re-fire every episode. "First ever" here means
        # "first episode in this training run the POLICY did this", not
        # "first time the game world permanently changed".
        self._flags_ever_fired = set()
        self._pokedex_seen_max = 0
        self._pokedex_owned_max = 0
        self._level_max = 0
        self._key_items_max = 0
        # (kind, key) -> number of EPISODES that fired this milestone, e.g.
        # ("flag", 1735) or ("pokedex_owned", 1). Read by the reward path:
        # milestone re-fires pay base / sqrt(1 + prior_fire_count) — the
        # same depleting-income rule as cell/map novelty. Without it, the
        # first milestone on the corridor is a full-price annuity every
        # episode and becomes a farming equilibrium (observed 2026-07-11:
        # the agent camped "talked_to_mom" at +500/episode for 200
        # episodes and never left the house). One increment per milestone
        # per episode, merged from the worker's pending set like cells.
        self._milestone_fire_counts = defaultdict(int)
        # bit_index -> number of EPISODES that fired (0->1) this event-flag bit,
        # over the WHOLE wEventFlags region (not just the curated milestone
        # table). Backs the optional event-novelty reward (rewards.py
        # _event_novelty_bonus): the first run-wide fire of a bit pays full,
        # re-fires decay by 1/sqrt(1 + prior_fire_count) — the same NGU rule
        # as cells/maps — so a genuinely-new interaction pays while a bit that
        # flips every episode (clock / sprite-visibility churn) is depleted to
        # ~0 within a handful of episodes. One increment per bit per episode,
        # merged from the worker's pending set like milestones.
        self._event_flag_fire_counts = defaultdict(int)
        # Durable checkpoint competence is tracked separately by episode
        # provenance. Seeded reaches are useful for learning a downstream
        # segment, but must never deplete the reward for consolidating the
        # same checkpoint from the canonical start.
        self._honest_checkpoint_fire_counts = defaultdict(int)
        self._seeded_checkpoint_fire_counts = defaultdict(int)

    def merge_visits(self, cells, maps):
        """Merge ONE episode's genuinely-visited cell/map keys into the
        canonical counts (+1 each). Cell keys may carry a fifth seeded flag;
        legacy four-part keys are treated as honest."""
        for raw_key in set(tuple(k) for k in (cells or [])):
            if len(raw_key) == 5:
                a, b, c, d, seeded = raw_key
            else:
                a, b, c, d = raw_key
                seeded = False
            key = (int(a), int(b), int(c), int(d))
            self._counts[key] += 1
            counts = self._seeded_counts if bool(seeded) else self._honest_counts
            counts[key] += 1
        for raw_key in set(tuple(k) for k in (maps or [])):
            if len(raw_key) == 3:
                a, b, seeded = raw_key
            else:
                a, b = raw_key
                seeded = False
            key = (int(a), int(b))
            self._map_counts[key] += 1
            counts = (
                self._seeded_map_counts if bool(seeded) else self._honest_map_counts
            )
            counts[key] += 1

    def merge_milestones(
        self,
        flags_fired,
        pokedex_seen_max,
        pokedex_owned_max,
        level_max,
        key_items_max,
        milestone_fires=(),
        event_flags_fired=(),
        checkpoint_fires=(),
        checkpoint_seeded=False,
    ):
        """Merge ONE episode's milestone ledger into the canonical run-wide
        state. Called by the agent with the pending values the worker
        reported in terminal_info (see Rewards.get_milestone_state).

        Returns True when anything actually changed, so the caller can
        mark the archive dirty and re-broadcast to the worker replicas.
        Without that signal, a milestone achieved after the CELL archive
        saturates (no new cells → nothing else sets the dirty flag) would
        never reach the workers, and the discovery-log dedup — which reads
        the worker replica — would silently re-log it every episode
        forever.
        """
        before = (
            len(self._flags_ever_fired),
            self._pokedex_seen_max,
            self._pokedex_owned_max,
            self._level_max,
            self._key_items_max,
        )
        self._flags_ever_fired.update(int(f) for f in (flags_fired or []))
        self._pokedex_seen_max = max(self._pokedex_seen_max, int(pokedex_seen_max))
        self._pokedex_owned_max = max(self._pokedex_owned_max, int(pokedex_owned_max))
        self._level_max = max(self._level_max, int(level_max))
        self._key_items_max = max(self._key_items_max, int(key_items_max))
        # One increment per (kind, key) per episode. Events arrive as
        # (kind, key) pairs but may have been list-ified in pipe transit.
        fired_any = False
        for event in milestone_fires or []:
            kind, key = event
            self._milestone_fire_counts[(str(kind), int(key))] += 1
            fired_any = True
        # One increment per event-flag bit per episode (see
        # _event_flag_fire_counts). A newly-fired bit also counts as a change
        # so the archive re-broadcasts (the reward decay reads the replica).
        for bit in event_flags_fired or []:
            self._event_flag_fire_counts[int(bit)] += 1
            fired_any = True
        checkpoint_counts = (
            self._seeded_checkpoint_fire_counts
            if checkpoint_seeded
            else self._honest_checkpoint_fire_counts
        )
        for flag_num in checkpoint_fires or []:
            checkpoint_counts[int(flag_num)] += 1
            fired_any = True
        after = (
            len(self._flags_ever_fired),
            self._pokedex_seen_max,
            self._pokedex_owned_max,
            self._level_max,
            self._key_items_max,
        )
        return fired_any or after != before

    def milestone_fire_count(self, kind, key):
        """Run-wide number of episodes that fired milestone (kind, key);
        0 if never fired. Non-mutating read (defaultdict[] would insert)."""
        return self._milestone_fire_counts.get((str(kind), int(key)), 0)

    def event_flag_fire_count(self, bit):
        """Run-wide number of episodes that fired event-flag ``bit`` (0->1);
        0 if never fired. Non-mutating read (backs the event-novelty decay)."""
        return self._event_flag_fire_counts.get(int(bit), 0)

    def checkpoint_fire_count(self, flag_num, seeded=False):
        """Completed episodes that fired a durable checkpoint by provenance."""
        counts = (
            self._seeded_checkpoint_fire_counts
            if seeded
            else self._honest_checkpoint_fire_counts
        )
        return counts.get(int(flag_num), 0)

    def flag_ever_fired(self, flag_num):
        return int(flag_num) in self._flags_ever_fired

    def pokedex_seen_max(self):
        return self._pokedex_seen_max

    def pokedex_owned_max(self):
        return self._pokedex_owned_max

    def level_max(self):
        return self._level_max

    def key_items_max(self):
        return self._key_items_max

    def to_state(self):
        """Serializable full-table state (pipes / torch.save)."""
        return {
            "cells": dict(self._counts),
            "honest_cell_counts": dict(self._honest_counts),
            "seeded_cell_counts": dict(self._seeded_counts),
            "maps": dict(self._map_counts),
            "honest_map_counts": dict(self._honest_map_counts),
            "seeded_map_counts": dict(self._seeded_map_counts),
            "flags_ever_fired": sorted(self._flags_ever_fired),
            "pokedex_seen_max": self._pokedex_seen_max,
            "pokedex_owned_max": self._pokedex_owned_max,
            "level_max": self._level_max,
            "key_items_max": self._key_items_max,
            "milestone_fire_counts": dict(self._milestone_fire_counts),
            "event_flag_fire_counts": dict(self._event_flag_fire_counts),
            "honest_checkpoint_fire_counts": dict(
                self._honest_checkpoint_fire_counts
            ),
            "seeded_checkpoint_fire_counts": dict(
                self._seeded_checkpoint_fire_counts
            ),
        }

    def load_state(self, state):
        """Replace both tables with a broadcast/checkpointed state."""
        self._counts = defaultdict(int)
        self._counts.update(
            {tuple(k): int(v) for k, v in (state.get("cells") or {}).items()}
        )
        self._honest_counts = defaultdict(int)
        self._seeded_counts = defaultdict(int)
        if "honest_cell_counts" in state or "seeded_cell_counts" in state:
            self._honest_counts.update(
                {
                    tuple(k): int(v)
                    for k, v in (state.get("honest_cell_counts") or {}).items()
                }
            )
            self._seeded_counts.update(
                {
                    tuple(k): int(v)
                    for k, v in (state.get("seeded_cell_counts") or {}).items()
                }
            )
        else:
            # Legacy archives did not record provenance. Treat their combined
            # history as honest so canonical-start decay remains continuous;
            # seeded history starts independently from zero.
            self._honest_counts.update(self._counts)
        self._map_counts = defaultdict(int)
        self._map_counts.update(
            {tuple(k): int(v) for k, v in (state.get("maps") or {}).items()}
        )
        self._honest_map_counts = defaultdict(int)
        self._seeded_map_counts = defaultdict(int)
        if "honest_map_counts" in state or "seeded_map_counts" in state:
            self._honest_map_counts.update(
                {
                    tuple(k): int(v)
                    for k, v in (state.get("honest_map_counts") or {}).items()
                }
            )
            self._seeded_map_counts.update(
                {
                    tuple(k): int(v)
                    for k, v in (state.get("seeded_map_counts") or {}).items()
                }
            )
        else:
            self._honest_map_counts.update(self._map_counts)
        self._flags_ever_fired = set(
            int(f) for f in (state.get("flags_ever_fired") or [])
        )
        self._pokedex_seen_max = int(state.get("pokedex_seen_max", 0))
        self._pokedex_owned_max = int(state.get("pokedex_owned_max", 0))
        self._level_max = int(state.get("level_max", 0))
        self._key_items_max = int(state.get("key_items_max", 0))
        self._milestone_fire_counts = defaultdict(int)
        self._milestone_fire_counts.update(
            {
                (str(k[0]), int(k[1])): int(v)
                for k, v in (state.get("milestone_fire_counts") or {}).items()
            }
        )
        self._event_flag_fire_counts = defaultdict(int)
        self._event_flag_fire_counts.update(
            {
                int(k): int(v)
                for k, v in (state.get("event_flag_fire_counts") or {}).items()
            }
        )
        self._honest_checkpoint_fire_counts = defaultdict(int)
        if (
            "honest_checkpoint_fire_counts" in state
            or "seeded_checkpoint_fire_counts" in state
        ):
            self._honest_checkpoint_fire_counts.update(
                {
                    int(k): int(v)
                    for k, v in (
                        state.get("honest_checkpoint_fire_counts") or {}
                    ).items()
                }
            )
        else:
            # Legacy checkpoints kept recordable flag counts only in the
            # generic milestone ledger. Preserve their depletion as honest;
            # provenance cannot be reconstructed after the fact.
            self._honest_checkpoint_fire_counts.update(
                {
                    int(key): int(count)
                    for (kind, key), count in self._milestone_fire_counts.items()
                    if kind == "flag" and is_recordable_checkpoint(key)
                }
            )
        self._seeded_checkpoint_fire_counts = defaultdict(int)
        self._seeded_checkpoint_fire_counts.update(
            {
                int(k): int(v)
                for k, v in (state.get("seeded_checkpoint_fire_counts") or {}).items()
            }
        )

    def map_count(self, map_bank, map_num, seeded=None):
        """Run-wide training entry count for (map_bank, map_num); 0 if never
        discovered. Non-mutating read (a defaultdict[] would insert zero
        entries and inflate n_cells_seen / to_state)."""
        if seeded is None:
            counts = self._map_counts
        else:
            counts = (
                self._seeded_map_counts if bool(seeded) else self._honest_map_counts
            )
        return counts.get((int(map_bank), int(map_num)), 0)

    def bank_count(self, map_bank, seeded=None):
        """Run-wide entry count for a whole bank (region): the sum of its
        maps' entry counts; 0 if the bank was never discovered. Non-mutating
        read. Used for bank-level novelty decay (how familiar the region is
        across the run)."""
        b = int(map_bank)
        if seeded is None:
            counts = self._map_counts
        else:
            counts = (
                self._seeded_map_counts if bool(seeded) else self._honest_map_counts
            )
        return sum(c for (mb, _mn), c in counts.items() if mb == b)

    def count(self, map_bank, map_num, x, y, seeded=None):
        """Completed-episode reach count for a cell.

        ``seeded`` selects an independent provenance ledger. ``None`` retains
        the aggregate compatibility view used by non-reward callers.
        """
        key = quantise(map_bank, map_num, x, y)
        if seeded is None:
            counts = self._counts
        else:
            counts = self._seeded_counts if bool(seeded) else self._honest_counts
        return counts.get(key, 0)

    def cell_key(self, map_bank, map_num, x, y):
        return quantise(map_bank, map_num, x, y)

    def n_cells_seen(self):
        return len(self._counts)

    def total_visits(self):
        return sum(self._counts.values())
