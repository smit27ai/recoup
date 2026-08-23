"""Tests for the measurement rig itself.

These matter more than any test of a strategy. If the rig is wrong, every number
Recoup reports is wrong in a direction nobody can see, and a confident wrong number
is worse than no number. `test_aa_null` is the load-bearing one.
"""

from __future__ import annotations

import statistics

import pytest

from recoup.domain import Arm
from recoup.generator.synthetic import ScenarioGenerator
from recoup.measure.harness import assign_arms, run, stratify
from recoup.policy.strategies import blind_retry, no_action, taxonomy_policy

SEEDS = range(1000, 1012)


@pytest.fixture(scope="module")
def scenarios() -> list:
    return [ScenarioGenerator(seed=s).generate(n_events=5000) for s in SEEDS]


def test_aa_null(scenarios: list) -> None:
    """A strategy that does nothing must measure as doing nothing.

    Both arms receive NO_ACTION, so the true lift is exactly 0.0 by construction.
    Any systematic departure means the estimator is inventing money. This is the
    test that caught the value-weighted estimator reporting +4.88%.
    """
    lifts = [
        run(s, no_action, "aa", seed=seed).lift for s, seed in zip(scenarios, SEEDS, strict=True)
    ]
    mean = statistics.fmean(lifts)
    assert abs(mean) < 0.01, f"A/A estimator is biased: mean lift {mean:+.2%}"
    assert max(abs(x) for x in lifts) < 0.05, f"A/A run out of tolerance: {lifts}"


def test_aa_beats_the_value_weighted_estimator(scenarios: list) -> None:
    """Guards the choice documented in the module docstring.

    If someone 'simplifies' back to value weighting, this fails and says why.
    """
    reports = [run(s, no_action, "aa", seed=seed) for s, seed in zip(scenarios, SEEDS, strict=True)]
    chosen = statistics.stdev([r.lift for r in reports])
    naive = statistics.stdev([r.naive_lift for r in reports])
    assert chosen < naive, f"count-rate sd {chosen:.2%} should beat value-weighted {naive:.2%}"


def test_real_strategy_separates_from_noise(scenarios: list) -> None:
    """The rig must be sensitive enough to see a genuinely better policy."""
    lifts = [
        run(s, taxonomy_policy, "p", seed=seed).lift
        for s, seed in zip(scenarios, SEEDS, strict=True)
    ]
    assert statistics.fmean(lifts) > 0.05, f"policy lift too small to detect: {lifts}"


def test_taxonomy_policy_beats_blind_retry(scenarios: list) -> None:
    """Knowing WHY a payment failed must be worth more than retrying everything."""
    for s, seed in zip(scenarios[:4], SEEDS, strict=False):
        smart = run(s, taxonomy_policy, "p", seed=seed).lift
        dumb = run(s, blind_retry, "r", seed=seed).lift
        assert smart > dumb, f"seed {seed}: taxonomy {smart:+.2%} vs blind {dumb:+.2%}"


def test_assignment_is_strategy_independent(scenarios: list) -> None:
    """Every strategy must be scored against the identical split, or the comparison
    is meaningless."""
    s = scenarios[0]
    a = {o.event_id: o.arm for o in run(s, no_action, "a").outcomes}
    b = {o.event_id: o.arm for o in run(s, blind_retry, "b").outcomes}
    assert a == b


def test_assignment_is_deterministic(scenarios: list) -> None:
    s = scenarios[0]
    assert assign_arms(s, seed=42) == assign_arms(s, seed=42)


def test_holdout_rate_is_honoured(scenarios: list) -> None:
    s = scenarios[0]
    arms = assign_arms(s, holdout_rate=0.2, seed=7)
    share = sum(1 for v in arms.values() if v is Arm.HOLDOUT) / len(arms)
    assert 0.18 < share < 0.22, share


def test_stratified_assignment_balances_value(scenarios: list) -> None:
    """The point of stratifying assignment: both arms carry a similar value profile."""
    s = scenarios[0]
    arms = assign_arms(s, seed=11)
    by_arm = {Arm.TREATMENT: [], Arm.HOLDOUT: []}
    for ev in s.events:
        by_arm[arms[ev.event_id]].append(ev.amount_paise)
    t = statistics.fmean(by_arm[Arm.TREATMENT])
    h = statistics.fmean(by_arm[Arm.HOLDOUT])
    assert abs(t - h) / t < 0.10, f"mean amount differs by {abs(t - h) / t:.1%}"


def test_holdout_is_never_contacted(scenarios: list) -> None:
    """A contacted holdout is a corrupted experiment."""
    for o in run(scenarios[0], taxonomy_policy, "p").outcomes:
        if o.arm is Arm.HOLDOUT:
            assert not o.was_contact
            assert o.executed.name == "NO_ACTION"


def test_each_stratum_is_homogeneous_in_root_cause(scenarios: list) -> None:
    """Strata are (root cause x amount quartile), so a stratum must never mix causes.

    Mixing them is exactly the failure that made amount-only stratification worse
    than no stratification at all -- see `stratify`.
    """
    from recoup.measure.harness import _cause_key

    s = scenarios[0]
    strata = stratify(s)
    causes: dict[int, set[str]] = {}
    for ev in s.events:
        causes.setdefault(strata[ev.event_id], set()).add(_cause_key(ev))
    for stratum, found in causes.items():
        assert len(found) == 1, f"stratum {stratum} mixes causes: {found}"


def test_stratification_beats_amount_only_on_aa(scenarios: list) -> None:
    """Locks in the finding. If someone reverts to amount-only strata, this fails."""
    amount_only = []
    chosen = []
    for s, seed in zip(scenarios, SEEDS, strict=True):
        ordered = sorted(s.events, key=lambda e: (e.amount_paise, e.event_id))
        size = max(1, len(ordered) // 40)
        by_amount = {ev.event_id: i // size for i, ev in enumerate(ordered)}
        amount_only.append(run(s, no_action, "aa", seed=seed, strata=by_amount).lift)
        chosen.append(run(s, no_action, "aa", seed=seed).lift)
    assert abs(statistics.fmean(chosen)) < abs(statistics.fmean(amount_only))


def test_no_strategy_beats_the_oracle_ceiling(scenarios: list) -> None:
    """A sanity bound. Exceeding it means the simulator or the scorer is broken."""
    for s, seed in zip(scenarios[:4], SEEDS, strict=False):
        r = run(s, taxonomy_policy, "p", seed=seed)
        assert r.incremental_paise <= r.ceiling_paise


def test_high_value_actions_are_queued_not_dropped(scenarios: list) -> None:
    """Regression: NEEDS_APPROVAL used to collapse into NO_ACTION, silently
    discarding the highest-value events with nothing in the metrics to show it."""
    from recoup.domain import ActionKind

    report = run(scenarios[0], taxonomy_policy, "p")
    queued = [o for o in report.outcomes if o.executed is ActionKind.QUEUED_FOR_APPROVAL]
    assert queued, "no events hit the approval threshold; the test is not exercising it"
    assert all(o.needed_approval for o in queued)
    assert report.queued_paise > 0
    # None of them may have been recorded as a plain decision not to act.
    assert all(o.executed is not ActionKind.NO_ACTION for o in queued)


def test_queued_money_is_not_counted_as_recovered(scenarios: list) -> None:
    """Parked money is visible but must never inflate the recovery number."""
    from recoup.domain import ActionKind

    report = run(scenarios[0], taxonomy_policy, "p")
    for o in report.outcomes:
        if o.executed is ActionKind.QUEUED_FOR_APPROVAL:
            assert not o.was_contact
