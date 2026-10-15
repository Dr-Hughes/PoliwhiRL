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
  2. Lets attention drift AWAY from a domain that has stopped producing any
     genuinely new (run-wide first-ever) progress for a while, and recover
     the moment it does again — a simple, conservative first version of a
     competence/learning-progress-adaptive weighting (the same idea behind
     this project's existing goal-ceiling entropy controller, generalised
     from "one coefficient" to "one coefficient per domain"). This is what
     makes the ordering of fundamentals-before-story EMERGENT rather than a
     hand-scheduled curriculum: early on every domain is producing fresh
     signal, so all stay near full weight; a domain that's exhausted its
     near-term novelty recedes, leaving room for whichever domain still has
     something left to teach.

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

DOMAINS = ("frontier", "battle", "milestone")

# Domain grouping of reward sources is applied directly in
# Rewards.calculate_reward (frontier: frontier/new_map/menu; battle: battle;
# milestone: flag+checkpoint/pokedex/key_item/level/event/map_goal/
# maps_visited). whiteout and step_penalty are direct, unscaled penalties —
# see the comment there for why (a rare, large, one-off penalty sharing a
# variance estimate with small frequent rewards would distort the scale for
# both).

# Diagnostic discovery "type" (see Rewards._discoveries_this_episode) ->
# domain, for the progress-activity signal. Battle has no discovery type of
# its own (no run-wide "first catch" entry exists), so it uses a separate,
# weaker per-episode proxy in Rewards.progressed_domains() instead.
_DISCOVERY_TYPE_DOMAIN = {
    "flag": "milestone",
    "pokedex_seen": "milestone",
    "pokedex_owned": "milestone",
    "level": "milestone",
    "key_item": "milestone",
    "map": "frontier",
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
        self._progress_window = int(
            config.get("domain_progress_window_episodes", 50)
        )
        self._floor = min(
            1.0, max(0.0, float(config.get("domain_progress_floor", 0.2)))
        )
        self._scales = {d: RunningMeanStd() for d in DOMAINS}
        self._episodes_since_progress = {d: 0 for d in DOMAINS}

    def note_episode_end(self, progressed_domains):
        """Call once per finished episode with the set of domain names that
        produced genuinely new progress this episode (see
        Rewards.progressed_domains)."""
        for d in DOMAINS:
            if d in progressed_domains:
                self._episodes_since_progress[d] = 0
            else:
                self._episodes_since_progress[d] += 1

    def activity_multiplier(self, domain):
        if self._progress_window <= 0:
            return 1.0
        stale = self._episodes_since_progress.get(domain, 0)
        frac = min(1.0, stale / self._progress_window)
        return 1.0 - frac * (1.0 - self._floor)

    def _typical_magnitude(self, domain):
        return max(abs(self._scales[domain].mean), self.min_magnitude)

    def scale(self, domain, raw_value):
        """Return raw_value rescaled to roughly unit typical-fire magnitude
        and weighted by the domain's current progress-activity. A no-op
        (returns raw_value unchanged) when disabled."""
        if not self.enabled or raw_value == 0.0:
            return raw_value
        rms = self._scales[domain]
        rms.update([raw_value])
        scale_factor = 1.0 / self._typical_magnitude(domain)
        return raw_value * scale_factor * self.activity_multiplier(domain)

    def diagnostics(self):
        """Current per-domain scale factor and activity multiplier, for
        logging/plots."""
        return {
            d: {
                "scale_factor": 1.0 / self._typical_magnitude(d),
                "activity": self.activity_multiplier(d),
                "episodes_since_progress": self._episodes_since_progress[d],
            }
            for d in DOMAINS
        }
