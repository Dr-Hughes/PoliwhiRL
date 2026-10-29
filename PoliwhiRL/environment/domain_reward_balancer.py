# -*- coding: utf-8 -*-
"""Cross-episode, per-DOMAIN reward balancing (opt-in, default off).

Motivation (2026-10-15 design discussion): every reward source currently
gets summed into one scalar and passed through a single shared scaler/clip.
A milestone fire (order 10-20 raw) and a frontier-cell fire (order 0.25 raw)
differ in typical magnitude by 40-80x, so "protect the fundamentals" has no
structural guarantee — it only works as well as the hand-tuned magnitudes
happen to balance, which this project's own history shows is fragile.

This module groups reward sources into a small number of general SKILL
DOMAINS (not per-stage, not per-area — see AGENTS.md's stated philosophy)
and, when enabled:

Domains (2026-10-16 addition): "catching" was split out of "milestone" so
that pushing pokedex/catch rewards hard (to drive toward gym badges' and
the wider Pokedex's reward surface as they become reachable) doesn't
dilute flag/checkpoint/event's shared scale factor — see the note on
``scale()`` below for why co-mingling magnitudes in one domain couples
them.

  1. Equalises TYPICAL FIRE MAGNITUDE across domains via a per-domain
     running-MEAN scale (only updated on non-zero fires, so the estimate
     reflects "how big is a fire" rather than being diluted by the many
     silent steps). Deliberately mean-based, not std-based: a domain whose
     fires are all close to the same size (e.g. a flat per-cell frontier
     bonus) has near-zero VARIANCE despite a very real, non-trivial
     magnitude — dividing by std would send a near-constant stream's scale
     factor toward infinity instead of toward "typical size ≈ 1". Dividing
     by the running mean magnitude gives the intended behaviour for both a
     near-constant stream and a variable one (a domain whose fires vary in
     size, e.g. escalating milestone thresholds, keeps its own internal
     spread after scaling — only the cross-domain typical scale is
     equalised). After scaling, a frontier fire and a milestone fire are
     both roughly unit-scale when they occur, so neither can drown the
     other out by raw size alone.
  2. Applies an OPTIONAL static per-domain priority weight
     (``domain_priority_weights``, default 1.0 for every domain) on top of
     the unit-magnitude normalisation above. This is the one durable "want
     this domain to matter more" lever: because step 1 already renormalises
     every domain toward the same typical per-fire size, raising a single
     source's raw config magnitude (e.g. ``pokedex_owned_reward``) is largely
     self-defeating once the running mean catches up — the domain's shared
     scale factor just shrinks to compensate, so the source converges back
     toward the same ~unit scale it would have reached anyway (see the note
     under ``DOMAINS`` for why magnitude bumps get normalised away). A
     priority weight instead multiplies the ALREADY-normalised value, so it
     holds for as long as it's configured rather than decaying back to
     parity — the correct way to say "weight catching 1.5x" or "weight
     milestones 2x" and have it actually stick.

  A progress-activity multiplier (domains that stopped producing run-wide
  first-ever progress receded toward a floor) lived here until 2026-10-23 and
  was removed: its per-domain progress proxies were asymmetric — "battle"
  counted as progressed whenever ANY battle reward fired in an episode while
  every other domain needed a run-wide first-ever discovery — so on any
  plateau it transferred reward weight onto wild-battle grinding, the one
  channel that can never go stale (observed in the open_world run: honest
  battle pinned at its episode cap from episode ~100 while event novelty
  depleted ~6x). Per-channel 1/sqrt(1+n) depletion already makes exhausted
  channels recede at the source; the coarser domain-level layer only added
  the asymmetry. ``Rewards.progressed_domains`` survives as a diagnostic.

Owned by PyBoyEnvironment (like VisitArchive) so it persists across
episodes within a worker process. Unlike VisitArchive it is NOT
broadcast/merged across workers — a per-worker-local approximation, reset
fresh each process start. This is a deliberate scope limit: the quantities
here are variance-stabilisation and attention-weighting aids, not
correctness-critical counters, so a per-worker estimate (~1/num_envs of the
run's data) is an acceptable trade against the complexity of cross-process
synchronisation.
"""

from PoliwhiRL.utils.running_stats import RunningMeanStd

DOMAINS = ("frontier", "battle", "milestone", "catching", "event")

# Domain grouping of reward sources is applied directly in
# Rewards.calculate_reward (frontier: frontier/new_map/menu; battle: battle;
# milestone: flag+checkpoint/key_item/level/event/map_goal/maps_visited;
# catching: pokedex seen+owned). whiteout and step_penalty are direct,
# unscaled penalties — see the comment there for why (a rare, large, one-off
# penalty sharing a variance estimate with small frequent rewards would
# distort the scale for both).
#
# Catching is its own domain (not folded into "milestone") specifically so
# it can be pushed hard — toward full Pokedex completion and gym badges as
# more of the map opens up — without diluting flag/checkpoint/event's shared
# scale factor. scale() estimates one typical-magnitude per domain from the
# combined per-step value handed to it (calculate_reward makes one
# domain_balancer.scale(domain, ...) call per domain per step with that
# domain's sources already summed); raising one source's raw magnitude
# inside a domain pulls that whole domain's running mean up, which shrinks
# the SCALED value of every other source sharing it. Splitting catching out
# means pokedex rewards can be raised freely without suppressing flag/badge/
# event's effective weight.

# Diagnostic discovery "type" (see Rewards._discoveries_this_episode) ->
# domain, for Rewards.progressed_domains (diagnostic only since the
# progress-activity multiplier was removed 2026-10-23). Battle has no
# discovery type of its own (no run-wide "first catch" entry exists), so it
# uses a separate, weaker per-episode proxy in Rewards.progressed_domains().
_DISCOVERY_TYPE_DOMAIN = {
    "flag": "milestone",
    "pokedex_seen": "catching",
    "pokedex_owned": "catching",
    "level": "milestone",
    "key_item": "milestone",
    "map": "frontier",
    "first_trainer_battle": "milestone",
    "first_trainer_battle_win": "battle",
}


def discovery_domain(discovery_type):
    return _DISCOVERY_TYPE_DOMAIN.get(discovery_type)


class DomainRewardBalancer:
    def __init__(self, config):
        self.enabled = bool(config.get("domain_reward_balancing_enabled", False))
        # Floor on the typical-magnitude estimate used for scaling — for a
        # domain that hasn't fired yet (or fires very small), this bounds
        # 1/magnitude so a near-zero mean can't blow the scale factor up.
        self.min_magnitude = float(config.get("domain_reward_min_magnitude", 0.05))
        # Hard clamp on the scale factor itself (2026-10-24 review finding 2:
        # per-fire normalisation turns "many small fires" into "many unit
        # fires", so delivered reward tracks fire COUNT, not just typical
        # size — on a long/deep episode this let the scale factor drift far
        # enough from parity that domain_scale_adjustment (scaled - raw,
        # summed) grew to the majority of delivered honest reward, which
        # means it's doing more of the shaping than the configured raw
        # prices are, and any new price added now gets renormalised away
        # before it can teach. Clamping bounds how far a domain's typical-
        # magnitude estimate can push its factor from parity in either
        # direction, in addition to (not instead of) min_magnitude above.
        self.scale_factor_min = float(config.get("domain_reward_scale_factor_min", 0.25))
        self.scale_factor_max = float(config.get("domain_reward_scale_factor_max", 4.0))
        self._scales = {d: RunningMeanStd() for d in DOMAINS}
        # Static per-domain priority multiplier applied after normalisation
        # (see module docstring point 3). Unset/missing domains default to
        # 1.0 — no-op unless a run deliberately opts a domain in.
        raw_weights = config.get("domain_priority_weights", {}) or {}
        self._priority_weights = {d: float(raw_weights.get(d, 1.0)) for d in DOMAINS}

    def _typical_magnitude(self, domain):
        return max(abs(self._scales[domain].mean), self.min_magnitude)

    def _scale_factor(self, domain):
        raw_factor = 1.0 / self._typical_magnitude(domain)
        return min(max(raw_factor, self.scale_factor_min), self.scale_factor_max)

    def _effective_multiplier(self, domain):
        # 2026-10-25 fix: the clamp above only bounded the pre-priority-weight
        # scale_factor, so a priority weight > 1.0 could still push the final
        # per-fire multiplier past scale_factor_max (observed live: frontier's
        # scale_factor sat near 3.67 of the 4.0 ceiling, and its 1.3 priority
        # weight pushed the actual multiplier applied to raw_value to 4.77,
        # silently exceeding the documented bound). Clamping the combined
        # product — not just its first factor — is what the module docstring
        # already claims this does.
        raw = self._scale_factor(domain) * self._priority_weights[domain]
        return min(max(raw, self.scale_factor_min), self.scale_factor_max)

    def scale(self, domain, raw_value):
        """Return raw_value rescaled to roughly unit typical-fire magnitude,
        then multiplied by the domain's static priority weight (default 1.0 —
        see module docstring point 2), with the combined multiplier clamped
        to [scale_factor_min, scale_factor_max] exactly like the unweighted
        scale factor. A no-op (returns raw_value unchanged) when disabled."""
        if not self.enabled or raw_value == 0.0:
            return raw_value
        rms = self._scales[domain]
        rms.update([raw_value])
        return raw_value * self._effective_multiplier(domain)

    def state_dict(self):
        """Serialize the dynamic (learned) state only — not config-derived
        fields like `min_magnitude`/`_priority_weights`, which come back
        from the config on the next construction regardless."""
        return {"scales": {d: self._scales[d].state_dict() for d in DOMAINS}}

    def load_state_dict(self, state):
        # Old checkpoints carry an "episodes_since_progress" key from the
        # removed progress-activity multiplier; it is ignored on purpose.
        scales = (state or {}).get("scales", {})
        for d in DOMAINS:
            if d in scales:
                self._scales[d].load_state_dict(scales[d])

    @staticmethod
    def merge_states(states):
        """Combine several workers' `state_dict()` snapshots (e.g. at
        checkpoint-save time, one per live worker process) into a single
        state suitable for seeding every worker's balancer on the next
        restart. Per-domain scale estimates are merged via the same Welford
        parallel-merge `RunningMeanStd` uses internally, so the combined
        estimate reflects all workers' fires, not just one."""
        states = [s for s in (states or []) if s]
        if not states:
            return {}
        merged_scales = {d: RunningMeanStd() for d in DOMAINS}
        for s in states:
            scales = s.get("scales", {})
            for d in DOMAINS:
                if d in scales:
                    merged_scales[d].merge_state_dict(scales[d])
        return {"scales": {d: merged_scales[d].state_dict() for d in DOMAINS}}

    def diagnostics(self):
        """Current per-domain scale factor, priority weight, and the
        resulting clamped effective multiplier actually applied in scale(),
        for logging/plots."""
        return {
            d: {
                "scale_factor": self._scale_factor(d),
                "priority_weight": self._priority_weights[d],
                "effective_multiplier": self._effective_multiplier(d),
            }
            for d in DOMAINS
        }
