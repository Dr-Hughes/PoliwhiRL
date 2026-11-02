# -*- coding: utf-8 -*-
"""The model's long-term notebook: short facts it chooses to keep (things NPCs
and signs said, what worked, what failed), carried across goals and attempts.

Unlike the per-turn ``notes`` scratchpad (replaced every turn), facts are only
ever appended; the notebook is capped at ``MAX_WORDS`` by dropping the
least-recalled, oldest facts. Search is plain keyword overlap — enough for a
few dozen facts, with no index to go stale. Every add, drop and recall is
appended to ``notebook.jsonl`` in the run directory.

Every fact remembers the goal attempt that wrote it. When the attempt ends,
its facts are settled: ``worked`` if the goal was reached, ``unconfirmed`` if
it was not. A model that is lost writes down why it is lost: ornith wrote
"Verified CHERRYGROVE origin is (27,15)" eight times across five failed
attempts, and every new attempt read it back as fact. Unconfirmed facts are
shown tagged and are the first to go when the notebook is full; claims of
certainty are stripped (only the harness can verify anything); and a new fact
that restates an old one replaces it instead of piling up beside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_WORDS = 500
MAX_FACT_WORDS = 40
# A new fact this similar (token Jaccard) to an old one restates it and
# replaces it. Ornith's eight "Verified ... origin (27,15)" variants sat at
# 0.6-0.8 to each other; the old skip-only threshold of 0.8 kept them all.
SUPERSEDE = 0.6
# Words that claim a certainty the model cannot have.
_CERTAIN = re.compile(
    r"\b(verified|confirmed|official(ly)?|proven|definitely|guaranteed)\b[:,]?\s*", re.IGNORECASE
)
_STOP = frozenset(
    "a an the to of in on at for and or is are was be it its i me my you your he she they "
    "this that with from by as if then so do does did not no yes can will just there here "
    "go get got have has had".split()
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9é]+", text.lower().replace("_", " "))
    return {w for w in words if w not in _STOP and len(w) > 1}


class Notebook:
    def __init__(self, path: Path | None = None):
        self.facts: list[dict] = []
        self._next_id = 1
        self._fh = open(path, "a") if path else None

    def _log(self, kind: str, **row):
        if self._fh:
            self._fh.write(json.dumps({"kind": kind, **row}, default=str) + "\n")
            self._fh.flush()

    def words(self) -> int:
        return sum(len(f["text"].split()) for f in self.facts)

    def add(self, texts, *, goal: str, map_name: str, turn: int, attempt: int = 0) -> list[dict]:
        """Append facts; skips empties, replaces a fact the new one restates.
        Returns the facts actually added."""
        added = []
        for raw in texts or []:
            said = " ".join(str(raw).split()[:MAX_FACT_WORDS]).strip()
            text = _CERTAIN.sub("", said).strip()
            if text != said:
                text = text[:1].upper() + text[1:]
            toks = _tokens(text)
            if not toks:
                continue
            old = next(
                (
                    f
                    for f in self.facts
                    if len(toks & f["tokens"]) / len(toks | f["tokens"]) >= SUPERSEDE
                ),
                None,
            )
            uses = 0
            if old is not None and old.get("status") == "worked":
                continue  # a fact from a goal that was reached outranks a restatement
            if old is not None:
                self.facts.remove(old)
                uses = old["uses"]
                self._log("supersede", id=old["id"], text=old["text"], turn=turn)
            fact = {
                "id": self._next_id,
                "text": text,
                "tokens": toks,
                "goal": goal,
                "map": map_name,
                "turn": turn,
                "uses": uses,
                "attempt": attempt,
                "status": "open",
            }
            self._next_id += 1
            self.facts.append(fact)
            added.append(fact)
            self._log(
                "add", id=fact["id"], text=text, goal=goal, map=map_name, turn=turn, attempt=attempt
            )
        while self.words() > MAX_WORDS and len(self.facts) > 1:
            victim = min(
                self.facts, key=lambda f: (f.get("status") != "unconfirmed", f["uses"], f["id"])
            )
            self.facts.remove(victim)
            self._log("drop", id=victim["id"], text=victim["text"], turn=turn)
        return added

    def settle(self, goal: str, attempt: int, success: bool) -> int:
        """Mark the facts this goal attempt wrote ``worked`` or
        ``unconfirmed``. Returns how many were settled."""
        status = "worked" if success else "unconfirmed"
        n = 0
        for f in self.facts:
            if f.get("status") == "open" and f["goal"] == goal and f.get("attempt") == attempt:
                f["status"] = status
                n += 1
        if n:
            self._log("settle", goal=goal, attempt=attempt, status=status, n=n)
        return n

    def search(
        self, query: str, k: int = 5, *, turn: int | None = None, bump: bool = True
    ) -> list[dict]:
        """Facts ranked by keyword overlap with ``query`` (ties: newest)."""
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for f in self.facts:
            overlap = len(q & (f["tokens"] | _tokens(f["map"]) | _tokens(f["goal"])))
            if overlap:
                scored.append((overlap, f["id"], f))
        hits = [f for _o, _i, f in sorted(scored, key=lambda t: (-t[0], -t[1]))[:k]]
        if bump:
            for f in hits:
                f["uses"] += 1
            if hits or turn is not None:
                self._log("recall", query=query, turn=turn, ids=[f["id"] for f in hits])
        return hits

    def export(self) -> list[dict]:
        return [{k: v for k, v in f.items() if k != "tokens"} for f in self.facts]

    def load(self, facts: list[dict]):
        """Restore facts saved by ``export`` (autosave / resume)."""
        for f in facts:
            fact = dict(f, tokens=_tokens(f["text"]))
            self.facts.append(fact)
            self._next_id = max(self._next_id, int(fact["id"]) + 1)
            self._log("restore", id=fact["id"], text=fact["text"])

    def since(self, turn: int) -> list[dict]:
        return [f for f in self.facts if f["turn"] >= turn]

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
