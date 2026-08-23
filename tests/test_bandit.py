"""Bandit and propensity tests.

The regressions worth pinning are the two failures found by measurement rather than
by reasoning: starvation-by-veto, and rewarding raw recovery instead of advantage.
Both looked completely healthy in the code and only showed up in the numbers.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from recoup.diagnosis.taxonomy import diagnose
from recoup.domain import ActionKind, AtRiskEvent, Channel, Customer, RiskKind
from recoup.generator.synthetic import ScenarioGenerator
from recoup.measure.harness import run
from recoup.policy.bandit import (
    INCENTIVE_BPS,
    RecoveryBandit,
    admissible,
    reward,
)
from recoup.policy.gates import IST, PolicyConfig
from recoup.policy.propensity import (
    PropensityModel,
    collect_training_log,
    featurise,
)
from recoup.policy.strategies import BanditStrategy, taxonomy_policy

NOW = datetime(2026, 9, 1, 11, 0, tzinfo=IST)


def _event(reason: str | None = "insufficient_funds", amount: int = 99_900, **kw) -> AtRiskEvent:
    base = {
        "event_id": "evt_0001",
        "customer_id": "cust_0001",
        "kind": RiskKind.FAILED_PAYMENT,
        "amount_paise": amount,
        "occurred_at": NOW,
        "error_reason": reason,
        "method": "card",
    }
    return AtRiskEvent(**{**base, **kw})


def _customer() -> Customer:
    return Customer(
        customer_id="cust_0001",
        segment="loyal",
        has_consent=True,
        on_dnd_registry=False,
        preferred_channel=Channel.WHATSAPP,
    )


def _dim() -> int:
    return len(featurise(_event(), None, _customer(), ActionKind.NO_ACTION))


# --- admissibility is a question of fact ------------------------------------


def test_expired_card_never_offers_a_retry() -> None:
    """Retrying is futile by construction. A bandit must not get to explore it."""
    options = admissible(_event("card_expired"), diagnose("card_expired"))
    assert not any(a.is_retry for a in options)
    assert ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH in options


def test_our_own_bug_offers_only_a_human() -> None:
    options = admissible(_event("invalid_order_id"), diagnose("invalid_order_id"))
    assert options == [ActionKind.ROUTE_TO_OPS]


def test_unmapped_code_offers_only_a_human() -> None:
    assert admissible(_event("who_knows_2027"), None) == [ActionKind.ROUTE_TO_OPS]


def test_gateway_failure_never_offers_contact() -> None:
    """Not the customer's fault, so there is nothing to tell them."""
    options = admissible(_event("gateway_technical_error"), diagnose("gateway_technical_error"))
    assert not any(a.is_contact for a in options)


def test_doing_nothing_is_always_an_option_when_anything_is() -> None:
    for reason in ("insufficient_funds", "card_expired", "authentication_failed"):
        assert ActionKind.NO_ACTION in admissible(_event(reason), diagnose(reason))


# --- starvation by veto (regression) ----------------------------------------


def test_unauthorised_incentive_is_not_offered() -> None:
    """The bug: a 15% incentive against 10% standing authority was always parked,
    never executed, so its arm kept maximal uncertainty and maximal exploration
    bonus -- forever. It was chosen on 3,052 of 5,000 events and produced 2 contacts."""
    tight = PolicyConfig(max_discount_bps=1000)
    options = admissible(_event("insufficient_funds"), diagnose("insufficient_funds"), tight)
    assert ActionKind.NUDGE_WITH_INCENTIVE not in options


def test_authorised_incentive_is_offered() -> None:
    loose = PolicyConfig(max_discount_bps=INCENTIVE_BPS)
    options = admissible(_event("insufficient_funds"), diagnose("insufficient_funds"), loose)
    assert ActionKind.NUDGE_WITH_INCENTIVE in options


def test_blocked_action_reduces_uncertainty_without_moving_the_estimate() -> None:
    """The general fix. A vetoed action must stop looking maximally attractive,
    but must not be blamed for an outcome that never happened."""
    bandit = RecoveryBandit(_dim())
    event, customer = _event("authentication_failed"), _customer()
    diagnosis = diagnose("authentication_failed")

    arm = bandit._arm(ActionKind.NUDGE)
    x = featurise(event, diagnosis, customer, ActionKind.NUDGE)
    _, bonus_before = arm.score(x, alpha=0.6)
    theta_before = arm.theta().copy()

    for _ in range(20):
        bandit.register_blocked(event, diagnosis, customer, ActionKind.NUDGE)

    _, bonus_after = arm.score(x, alpha=0.6)
    assert bonus_after < bonus_before, "uncertainty must fall"
    assert (arm.theta() == theta_before).all(), "reward estimate must not move"


def test_systematically_blocked_action_stops_being_chosen() -> None:
    """End to end: an action that is always vetoed must not dominate selection."""
    bandit = RecoveryBandit(_dim(), prior=None)
    event, customer = _event("authentication_failed"), _customer()
    diagnosis = diagnose("authentication_failed")

    for _ in range(200):
        choice = bandit.select(event, diagnosis, customer)
        if choice.action.is_contact:
            bandit.register_blocked(event, diagnosis, customer, choice.action)
        else:
            bandit.update(event, diagnosis, customer, choice.action, recovered=True)

    final = [bandit.select(event, diagnosis, customer).action for _ in range(10)]
    assert not all(a.is_contact for a in final), "blocked action still dominates"


# --- advantage, not raw recovery (regression) -------------------------------


def test_doing_nothing_scores_zero_advantage() -> None:
    """By construction. Every other action has to beat it on causation, not on
    correlation with recovery."""
    bandit = RecoveryBandit(_dim())
    choice = bandit.select(
        _event("insufficient_funds"), diagnose("insufficient_funds"), _customer()
    )
    if choice.action is ActionKind.NO_ACTION:
        assert choice.expected_reward == 0.0


def test_high_self_heal_does_not_make_doing_nothing_look_good() -> None:
    """The bug: rewarding raw recovery meant NO_ACTION scored 0.52 on a GATEWAY_DOWN
    failure that self-heals 52% of the time -- chasing gross, not lift. The baseline
    arm learns that 0.52 and everything else is measured against it."""
    bandit = RecoveryBandit(_dim())
    event, customer = _event("gateway_technical_error"), _customer()
    diagnosis = diagnose("gateway_technical_error")

    for i in range(60):
        bandit.update(event, diagnosis, customer, ActionKind.NO_ACTION, recovered=i % 2 == 0)
    baseline = bandit._baseline(event, diagnosis, customer)
    assert baseline > 0.2, f"baseline should learn the self-heal rate, got {baseline}"

    # An action that recovers at the same rate as doing nothing has zero advantage.
    for i in range(40):
        bandit.update(event, diagnosis, customer, ActionKind.RETRY_NOW, recovered=i % 2 == 0)
    x = featurise(event, diagnosis, customer, ActionKind.RETRY_NOW)
    estimate = float(bandit._arm(ActionKind.RETRY_NOW).theta() @ x)
    assert abs(estimate) < 0.35, f"no-better-than-nothing must score near zero, got {estimate}"


def test_reward_penalises_contact() -> None:
    assert reward(True, ActionKind.NUDGE) < reward(True, ActionKind.RETRY_NOW)
    assert reward(True, ActionKind.RETRY_NOW) == 1.0


# --- value-aware exploration ------------------------------------------------


def test_cheap_events_explore_and_expensive_ones_do_not() -> None:
    """Information from one expensive event is worth no more than from a cheap one;
    the regret is worth 900x more."""
    bandit = RecoveryBandit(_dim())
    diagnosis = diagnose("insufficient_funds")
    cheap = bandit.select(_event("insufficient_funds", amount=9_900), diagnosis, _customer())
    dear = bandit.select(_event("insufficient_funds", amount=90_000_00), diagnosis, _customer())
    assert cheap.uncertainty > 0.0
    assert dear.uncertainty == 0.0
    assert not dear.explored


def test_single_option_needs_no_exploration() -> None:
    choice = RecoveryBandit(_dim()).select(
        _event("invalid_order_id"), diagnose("invalid_order_id"), _customer()
    )
    assert choice.action is ActionKind.ROUTE_TO_OPS
    assert choice.uncertainty == 0.0


# --- propensity model -------------------------------------------------------


@pytest.fixture(scope="module")
def fitted() -> tuple[PropensityModel, list]:
    train = ScenarioGenerator(seed=1).generate(n_events=4000, n_customers=600)
    test = ScenarioGenerator(seed=2).generate(n_events=2000, n_customers=400)
    model = PropensityModel().fit(collect_training_log(train, seed=1))
    return model, collect_training_log(test, seed=2)


def test_model_is_calibrated_on_held_out_data(fitted) -> None:
    """Calibration, not AUC. The bandit multiplies these by rupees, so a model that
    ranks well but is overconfident makes it systematically over-act."""
    model, held_out = fitted
    assert model.reliability(held_out).ece < 0.06


def test_model_separates_futile_from_useful_actions(fitted) -> None:
    """The floor: retrying an expired card must score below switching instrument."""
    model, _ = fitted
    event, customer = _event("card_expired"), _customer()
    diagnosis = diagnose("card_expired")
    retry = model.predict(event, diagnosis, customer, ActionKind.RETRY_NOW)
    switch = model.predict(event, diagnosis, customer, ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH)
    assert switch > retry


def test_unfitted_model_refuses_to_predict() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PropensityModel().predict(_event(), None, _customer(), ActionKind.NUDGE)


def test_fitting_needs_enough_data() -> None:
    with pytest.raises(ValueError, match="at least 50"):
        PropensityModel().fit([])


def test_training_log_covers_every_action() -> None:
    """A log from a good policy would never contain evidence that retrying an
    expired card is futile, because a good policy never does it."""
    scenario = ScenarioGenerator(seed=3).generate(n_events=1500, n_customers=300)
    log = collect_training_log(scenario, seed=3)
    assert len(log) == 1500
    assert 0 < sum(1 for r in log if r.recovered) < len(log)


# --- against the deterministic policy ---------------------------------------


def test_bandit_beats_the_naive_baselines() -> None:
    """It need not beat taxonomy_policy -- see the module docstring -- but a learned
    policy that loses to blind retry would not be worth shipping at all."""
    from recoup.policy.strategies import blast, blind_retry

    scenario = ScenarioGenerator(seed=20260905).generate(n_events=5000)
    bandit = run(scenario, BanditStrategy(RecoveryBandit(_dim())), "bandit").lift
    assert bandit > run(scenario, blind_retry, "retry").lift
    assert bandit > run(scenario, blast, "blast").lift


def test_bandit_uses_fewer_contacts_than_the_deterministic_policy() -> None:
    """The contact penalty is doing its job."""
    scenario = ScenarioGenerator(seed=20260905).generate(n_events=5000)
    bandit = run(scenario, BanditStrategy(RecoveryBandit(_dim())), "bandit")
    fixed = run(scenario, taxonomy_policy, "taxonomy")
    assert bandit.contacts < fixed.contacts
