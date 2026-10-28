"""Reward verification: detect game events from RAM transitions and check that
the training reward function paid the sources it should have.

Each env step the detector diffs the previous and current RAM read and emits
events (first entry to a map, story flag set, battle start/end, catch, level
up, ...). Every event type has an expectation rule naming the reward sources
that should pay within a short window after it:

  must  — the source is designed to pay for this event; a miss is a finding
  may   — the source pays for it only under conditions (novelty, caps,
          run-wide decay), so the hit rate is reported, not judged

Independently, any payout of a milestone-type source of at least
``UNEXPLAINED_MIN`` with no event in its window is flagged as unexplained:
that is how a farm or a mis-wired channel shows up.

The rules mirror the sources in ``Rewards.get_episode_breakdown`` as of
2026-10-28; read rewards.py for each source's real conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PoliwhiRL.checkpoints import DERIVED_FLAG_TABLE

DERIVED = dict(DERIVED_FLAG_TABLE)

# event kind -> (strength, sources any of which counts as paid, window in steps)
RULES = {
    "map_first_entry": ("must", ("new_map",), 3),
    "story_checkpoint": ("must", ("checkpoint_honest", "checkpoint_seeded", "flag"), 4),
    "event_flag": ("may", ("event",), 4),
    "battle_start": ("may", ("battle", "first_trainer_battle"), 6),
    "battle_won": ("may", ("battle", "first_trainer_battle_win"), 12),
    "whiteout": ("must", ("whiteout",), 30),
    "catch": ("must", ("catch", "pokedex"), 12),
    "pokedex_seen": ("may", ("pokedex",), 8),
    "level_up": ("must", ("level",), 12),
    "key_item": ("must", ("key_item",), 6),
    "talk": ("may", ("talk",), 4),
}
# Sources that pay as continuous shaping; never "unexplained".
SHAPING = {"frontier", "frontier_potential", "resource_potential", "menu", "step_penalty",
           "rounding_adjustment", "clip_adjustment", "delivery_cast_adjustment",
           "domain_scale_adjustment", "delivered_total", "maps_visited", "map_rung"}
UNEXPLAINED_MIN = 1.0


def _flag_bits(flags) -> set[int]:
    out = set()
    for i, byte in enumerate(flags):
        b = int(byte)
        while b:
            low = b & -b
            out.add(i * 8 + low.bit_length() - 1)
            b ^= low
    return out


@dataclass
class Event:
    step: int
    kind: str
    detail: str
    goal: str
    attempt: int
    context: dict = field(default_factory=dict)
    paid: dict = field(default_factory=dict)
    verdict: str = "pending"
    # Step the payout window counts from. new_map is deferred while a scene
    # runs on the newly entered map, so that event's window slides forward
    # until the scene ends.
    anchor: int | None = None

    def __post_init__(self):
        if self.anchor is None:
            self.anchor = self.step

    def row(self) -> dict:
        return {"step": self.step, "kind": self.kind, "detail": self.detail, "goal": self.goal,
                "attempt": self.attempt, "context": self.context, "paid": self.paid,
                "verdict": self.verdict}


class RewardVerifier:
    """Feed it every step; it emits Events and later fills in what paid."""

    def __init__(self, on_event=None):
        self.on_event = on_event or (lambda ev: None)
        self.events: list[Event] = []
        self.unexplained: list[dict] = []
        self._open: list[Event] = []
        self._recent: list[tuple[int, dict]] = []  # (step, sources) last ~32 steps
        self.new_episode()

    def new_episode(self):
        """Call when the env starts a new episode (a seeded restart): the
        reward function's per-episode novelty state resets with it."""
        self._prev = None
        self._maps = set()
        self._battle = None
        self._start_bits = set()

    # -- per step -------------------------------------------------------------
    def step(self, step: int, goal: str, attempt: int, button: str, v: dict, game, sources: dict):
        prev = self._prev
        new: list[Event] = []

        def emit(kind, detail, **ctx):
            new.append(Event(step, kind, detail, goal, attempt, ctx))

        mp = (int(v["map_bank"]), int(v["map_num"]))
        if prev is None:
            self._maps.add(mp)
            self._start_bits = _flag_bits(v["story_flags"])
        else:
            if mp not in self._maps:
                self._maps.add(mp)
                from tools.llm_agent.game import map_name
                emit("map_first_entry", map_name(*mp), scene_active=bool(v["script_active"]),
                     map=mp)
            gained = _flag_bits(v["story_flags"]) - _flag_bits(prev["story_flags"])
            for bit in sorted(gained):
                # The flag reward pays a table flag only on 0 -> 1 relative to
                # the episode start, so one already set then (1726 is set from
                # game start, cleared by the egg and set again after the
                # rival fight) is not a checkpoint event.
                if bit in DERIVED and bit not in self._start_bits:
                    emit("story_checkpoint", f"flag {bit} {DERIVED[bit]}")
                emit("event_flag", f"flag {bit}")
            bt, pbt = int(v["battle_type"]), int(prev["battle_type"])
            if bt and not pbt:
                self._battle = {"type": bt, "enemy_hp": int(v["enemy_hp"]), "start": step,
                                "party_hp": sum(m["hp"] for m in game.party())}
                emit("battle_start", "wild" if bt == 1 else "trainer")
            if bt and self._battle is not None:
                # A win is the enemy's HP reaching 0 (what rewards.py pays
                # on), not the battle screen closing, which comes 10+ steps
                # later after the EXP text.
                if self._battle["enemy_hp"] > 0 and int(v["enemy_hp"]) == 0:
                    emit("battle_won", "wild" if bt == 1 else "trainer",
                         length=step - self._battle["start"])
                self._battle["enemy_hp"] = int(v["enemy_hp"])
            if pbt and not bt and self._battle is not None:
                party_hp = sum(m["hp"] for m in game.party())
                if self._battle["enemy_hp"] != 0 and (
                        party_hp == 0 or mp != (int(prev["map_bank"]), int(prev["map_num"]))):
                    emit("whiteout", "party fainted")
                self._battle = None
            if int(v["pokedex_owned"]) > int(prev["pokedex_owned"]):
                emit("catch", f"pokedex owned {prev['pokedex_owned']} -> {v['pokedex_owned']}")
            if int(v["pokedex_seen"]) > int(prev["pokedex_seen"]):
                emit("pokedex_seen", f"seen {prev['pokedex_seen']} -> {v['pokedex_seen']}")
            lvl, plvl = int(v["party_info"][1]), int(prev["party_info"][1])
            if lvl > plvl and int(v["party_info"][0]) == int(prev["party_info"][0]):
                emit("level_up", f"total party level {plvl} -> {lvl}")
            if int(v["key_items_count"]) > int(prev["key_items_count"]):
                emit("key_item", f"key items {prev['key_items_count']} -> {v['key_items_count']}")
            if (button == "a" and v["script_active"] and not prev["script_active"]
                    and not bt):
                emit("talk", "started a conversation/interaction")
        self._prev = v

        # attribute this step's payouts to open events, close expired ones
        for ev in self._open + new:
            if (ev.kind == "map_first_entry" and v["script_active"]
                    and ev.context.get("map") == mp):
                ev.anchor = step
            if step - ev.anchor <= RULES[ev.kind][2]:
                for src, val in sources.items():
                    if src in RULES[ev.kind][1]:
                        ev.paid[src] = round(ev.paid.get(src, 0.0) + val, 4)
        for ev in new:
            self.events.append(ev)
            self._open.append(ev)
        still = []
        for ev in self._open:
            if step - ev.anchor >= RULES[ev.kind][2]:
                self._close(ev)
            else:
                still.append(ev)
        self._open = still

        # unexplained milestone-type payouts
        self._recent.append((step, sources))
        self._recent = self._recent[-40:]
        for src, val in sources.items():
            if src in SHAPING or abs(val) < UNEXPLAINED_MIN:
                continue
            explained = any(
                src in RULES[ev.kind][1] and 0 <= step - ev.anchor <= RULES[ev.kind][2]
                for ev in self.events[-50:]
            )
            if not explained:
                self.unexplained.append({"step": step, "goal": goal, "attempt": attempt,
                                         "source": src, "value": val, "button": button})
        return new

    def _close(self, ev: Event):
        strength, _srcs, _w = RULES[ev.kind]
        hit = any(v > 0 or (ev.kind == "whiteout" and v < 0) for v in ev.paid.values())
        if hit:
            ev.verdict = "paid"
        elif strength == "must":
            ev.verdict = "MISSED"
        else:
            ev.verdict = "not paid (conditional)"
        self.on_event(ev)

    def flush(self):
        for ev in self._open:
            self._close(ev)
        self._open = []

    # -- summaries -----------------------------------------------------------------
    def table(self, goal: str | None = None, attempt: int | None = None) -> list[dict]:
        rows = {}
        for ev in self.events:
            if goal is not None and (ev.goal != goal or (attempt and ev.attempt != attempt)):
                continue
            r = rows.setdefault(ev.kind, {"kind": ev.kind, "strength": RULES[ev.kind][0],
                                          "sources": "/".join(RULES[ev.kind][1]),
                                          "events": 0, "paid": 0, "missed": []})
            r["events"] += 1
            if ev.verdict == "paid":
                r["paid"] += 1
            elif ev.verdict != "pending":
                r["missed"].append(ev)
        return list(rows.values())
