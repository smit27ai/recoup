"""Measurement harness: run a strategy over a scenario and report honest numbers.

The whole point of this module is the gap between two columns:

  GROSS       every rupee that came in on an event we touched
  INCREMENTAL what came in BECAUSE we touched it, measured against a randomised
              holdout of otherwise-identical events

Gross is what gets put on dunning-vendor landing pages. Incremental is what a
finance team would actually pay for. They differ by the self-heal bucket, which on
this population is roughly a fifth of all at-risk money.

Holdout design: assignment is per EVENT, seeded, and independent of the strategy, so
every strategy sees the same split and the comparison is apples to apples. Holdout
events get NO_ACTION regardless of what the strategy wanted, and their outcome is
drawn from self-heal alone. That is the counterfactual.

Estimator choice, and why it is not the obvious one
---------------------------------------------------
At-risk portfolios are brutally skewed: on the default population the top 1% of
events carry ~26% of all at-risk rupees. The obvious estimator -- difference in
VALUE-weighted recovery rate between arms -- is wrecked by that tail. An A/A test
(`no_action`, whose lift must be exactly zero by construction) measured +4.88% under
it, and at n=60k it produced -4.44% with a bootstrap CI that EXCLUDED zero: a
confident false positive, the worst failure mode a measurement rig has.

Two fixes were tried against 12 seeded A/A replications:

  estimator                          A/A mean     sd    detects real policy lift
  value-weighted, pooled              +1.87%    3.81%          +4.31%
  value-weighted, 40 strata           -1.49%    4.50%          +0.97%
  count-rate, 40 strata               -0.14%    3.86%          +2.33%
  count-rate, pooled  <-- chosen      -0.18%    1.24%         +10.85%

Stratifying the ESTIMATOR made things worse, which was the opposite of the
expectation. Splitting into buckets means each bucket rate comes from few events,
and value-weighting those noisy per-bucket estimates injects more variance than the
tail imbalance it removes. At this sample size variance dominates bias, so the
pooled count-rate estimator wins on both -- it is nearly unbiased AND has a third
the spread, which is why it separates a real strategy from noise at all.

So: lift is estimated as a pooled difference in per-event recovery PROBABILITY, then
converted to money by multiplying through total at-risk value. That step assumes the
lift is not correlated with amount. On this population it is not (lift is driven by
root cause and segment). On a real portfolio it might be, and then the stratified
estimator becomes the correct one -- `stratified_lift()` is kept for exactly that,
and the crossover is a sample-size question, not a taste question.

Assignment is stratified on (root cause x amount quartile) -- see `stratify`, which
documents why stratifying on amount alone made things measurably worse.
`test_aa_null` guards the whole rig permanently: if it fails, no other number this
module prints can be trusted.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil

from recoup.diagnosis.taxonomy import diagnose
from recoup.domain import ActionKind, Arm, AtRiskEvent, RiskKind
from recoup.generator.synthetic import Scenario
from recoup.policy.gates import (
    CustomerState,
    Disposition,
    EventState,
    GateContext,
    PolicyConfig,
    ProposedAction,
    evaluate,
)
from recoup.policy.strategies import Strategy


@dataclass(frozen=True, slots=True)
class Outcome:
    """One event, one decision, one result. This is what the audit ledger stores."""

    event_id: str
    stratum: int
    arm: Arm
    intended: ActionKind
    executed: ActionKind
    """Differs from `intended` whenever a gate vetoed it."""
    gate_denials: tuple[str, ...]
    needed_approval: bool
    recovered: bool
    amount_paise: int
    was_contact: bool


@dataclass(frozen=True, slots=True)
class Report:
    strategy: str
    outcomes: tuple[Outcome, ...]
    scenario_total_paise: int
    ceiling_paise: int
    self_heal_paise: int

    def _arm(self, arm: Arm) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.arm is arm)

    @staticmethod
    def _rate(rows: Sequence[Outcome]) -> float:
        """Recovered rupees per at-risk rupee. Value-weighted, not count-weighted --
        recovering ten Rs.99 subscriptions is not the same win as one Rs.90,000 invoice."""
        at_risk = sum(o.amount_paise for o in rows)
        if at_risk == 0:
            return 0.0
        return sum(o.amount_paise for o in rows if o.recovered) / at_risk

    def _by_stratum(self) -> dict[int, tuple[list[Outcome], list[Outcome]]]:
        buckets: dict[int, tuple[list[Outcome], list[Outcome]]] = defaultdict(lambda: ([], []))
        for o in self.outcomes:
            treat, hold = buckets[o.stratum]
            (treat if o.arm is Arm.TREATMENT else hold).append(o)
        return dict(buckets)

    @staticmethod
    def _count_rate(rows: Sequence[Outcome]) -> float:
        """Share of events recovered. Variance is p(1-p)/n regardless of amount size,
        which is precisely why this survives the heavy tail and `_rate` does not."""
        return sum(1 for o in rows if o.recovered) / len(rows) if rows else 0.0

    def stratified_lift(self) -> float:
        """Value-weighted average of within-stratum count-rate lifts.

        NOT the default -- see the module docstring. Correct when lift is correlated
        with amount and there are enough events per stratum to estimate a rate; at
        n=5000 it has 3x the spread of the pooled estimator and buys nothing.
        """
        buckets = self._by_stratum()
        total = sum(sum(o.amount_paise for o in treat + hold) for treat, hold in buckets.values())
        if total == 0:
            return 0.0
        lift = 0.0
        for treat, hold in buckets.values():
            if not treat or not hold:
                continue  # a stratum with an empty arm carries no information
            weight = sum(o.amount_paise for o in treat + hold) / total
            lift += weight * (self._count_rate(treat) - self._count_rate(hold))
        return lift

    @property
    def gross_paise(self) -> int:
        return sum(o.amount_paise for o in self._arm(Arm.TREATMENT) if o.recovered)

    @property
    def treatment_rate(self) -> float:
        return self._count_rate(self._arm(Arm.TREATMENT))

    @property
    def holdout_rate(self) -> float:
        return self._count_rate(self._arm(Arm.HOLDOUT))

    @property
    def naive_lift(self) -> float:
        """The value-weighted estimator that failed the A/A test. Kept so the tests
        can assert it is worse, and so nobody reintroduces it thinking it is obvious."""
        return self._rate(self._arm(Arm.TREATMENT)) - self._rate(self._arm(Arm.HOLDOUT))

    @property
    def lift(self) -> float:
        return self.treatment_rate - self.holdout_rate

    @property
    def incremental_paise(self) -> int:
        """Lift extrapolated across all at-risk money. The number that counts."""
        return int(self.lift * self.scenario_total_paise)

    @property
    def contacts(self) -> int:
        return sum(1 for o in self.outcomes if o.was_contact)

    @property
    def blocked(self) -> int:
        return sum(1 for o in self.outcomes if o.gate_denials)

    @property
    def approvals(self) -> int:
        return sum(1 for o in self.outcomes if o.needed_approval)

    @property
    def queued_paise(self) -> int:
        """Money parked awaiting a human. Not lost, not recovered -- visible."""
        return sum(
            o.amount_paise for o in self.outcomes if o.executed is ActionKind.QUEUED_FOR_APPROVAL
        )

    @property
    def paise_per_contact(self) -> float:
        return self.incremental_paise / self.contacts if self.contacts else 0.0

    def lift_ci(self, iterations: int = 400, seed: int = 7) -> tuple[float, float]:
        """Bootstrap 95% CI on the lift, resampling exactly as the estimator computes.

        The CI is conditional on the realised assignment -- it captures outcome noise,
        not the luck of the split. That limitation is why `test_aa_null` runs across
        many SEEDS rather than trusting one run CI: a single interval cannot tell you
        your randomisation was unlucky, which is how the earlier estimator managed to
        exclude zero on an A/A test.
        """
        rng = random.Random(seed)
        treat, hold = self._arm(Arm.TREATMENT), self._arm(Arm.HOLDOUT)
        if not treat or not hold:
            return (0.0, 0.0)
        samples = []
        for _ in range(iterations):
            t = [rng.choice(treat) for _ in treat]
            h = [rng.choice(hold) for _ in hold]
            samples.append(self._count_rate(t) - self._count_rate(h))
        samples.sort()
        lo = samples[int(0.025 * len(samples))]
        hi = samples[int(0.975 * len(samples)) - 1]
        return (lo, hi)

    @property
    def capture_of_contestable(self) -> float:
        """Share of the genuinely winnable money this strategy actually won."""
        contestable = self.ceiling_paise - self.self_heal_paise
        return self.incremental_paise / contestable if contestable > 0 else 0.0


N_STRATA = 4
"""Equal-count amount buckets, crossed with root cause. See `stratify`."""


def _cause_key(event: AtRiskEvent) -> str:
    """The covariate that actually drives recovery probability."""
    if event.kind is RiskKind.CHECKOUT_ABANDONED:
        return "AUTH_ABANDONED"
    if event.kind is RiskKind.INVOICE_OVERDUE:
        return "FUNDS"
    diag = diagnose(event.error_reason)
    return str(diag.root_cause) if diag else "UNMAPPED"


def stratify(scenario: Scenario, n_strata: int = N_STRATA) -> dict[str, int]:
    """Bucket events by (root cause x amount quartile). Pure function of the scenario.

    Stratifying on AMOUNT ALONE was the intuitive choice and it made the A/A test
    WORSE (-1.06% vs -0.18% for no stratification at all). The reason is that amount
    scales how much a recovery is worth but says nothing about whether it happens;
    the variable that drives the outcome is root cause. Balancing arms on the money
    while leaving the outcome driver unbalanced is worse than not balancing at all,
    because it buys a false sense of rigour.

    Measured across 12 seeded A/A replications:

      assignment strata     A/A mean      sd   max|err|   detects policy lift
      none                    -0.18%   1.24%      3.05%          +10.85%
      amount only             -1.06%   1.39%      3.40%          +10.00%
      root cause only         -0.11%   1.35%      1.90%          +10.87%
      cause x amount           -0.02%   1.29%      2.36%          +11.06%   <-- chosen

    Cause x amount balances the outcome driver AND the money, and is best on both
    bias and detection. The rule generalises: stratify on what predicts the OUTCOME
    first, and on what scales the value second.
    """
    ordered = sorted(scenario.events, key=lambda e: (e.amount_paise, e.event_id))
    size = max(1, ceil(len(ordered) / n_strata))
    amount_bucket = {ev.event_id: i // size for i, ev in enumerate(ordered)}

    ids: dict[tuple[str, int], int] = {}
    return {
        ev.event_id: ids.setdefault((_cause_key(ev), amount_bucket[ev.event_id]), len(ids))
        for ev in scenario.events
    }


def assign_arms(
    scenario: Scenario,
    holdout_rate: float = 0.20,
    seed: int = 20260905,
    strata: dict[str, int] | None = None,
) -> dict[str, Arm]:
    """Stratified holdout assignment, balanced within amount stratum.

    Within each stratum we shuffle and take exactly `holdout_rate` of the events, so
    both arms carry a near-identical value profile. Independent of the strategy by
    construction -- every strategy is scored against the same split.
    """
    rng = random.Random(seed)
    buckets: dict[int, list[str]] = defaultdict(list)
    strata = strata or stratify(scenario)
    for ev in sorted(scenario.events, key=lambda e: e.event_id):
        buckets[strata[ev.event_id]].append(ev.event_id)

    arms: dict[str, Arm] = {}
    for members in buckets.values():
        rng.shuffle(members)
        n_hold = round(len(members) * holdout_rate)
        for i, eid in enumerate(members):
            arms[eid] = Arm.HOLDOUT if i < n_hold else Arm.TREATMENT
    return arms


def run(
    scenario: Scenario,
    strategy: Strategy,
    name: str,
    holdout_rate: float = 0.20,
    seed: int = 20260905,
    config: PolicyConfig | None = None,
    n_strata: int = N_STRATA,
    strata: dict[str, int] | None = None,
) -> Report:
    cfg = config or PolicyConfig()
    strata = strata if strata is not None else stratify(scenario, n_strata)
    arms = assign_arms(scenario, holdout_rate=holdout_rate, seed=seed, strata=strata)
    outcome_rng = random.Random(seed + 1)  # coin flips for recovery
    contacts_seen: dict[str, list[datetime]] = {}
    outcomes: list[Outcome] = []

    for ev in scenario.events:
        gt = scenario.truth[ev.event_id]
        cust = scenario.customers[ev.customer_id]
        arm = arms[ev.event_id]

        if arm is Arm.HOLDOUT:
            recovered = outcome_rng.random() < gt.self_heal_probability
            outcomes.append(
                Outcome(
                    event_id=ev.event_id,
                    stratum=strata[ev.event_id],
                    arm=arm,
                    intended=ActionKind.NO_ACTION,
                    executed=ActionKind.NO_ACTION,
                    gate_denials=(),
                    needed_approval=False,
                    recovered=recovered,
                    amount_paise=ev.amount_paise,
                    was_contact=False,
                )
            )
            continue

        diag = diagnose(ev.error_reason)
        intended = strategy(ev, diag, cust)

        prior = contacts_seen.setdefault(cust.customer_id, [])
        verdict = evaluate(
            GateContext(
                action=ProposedAction(
                    event_id=ev.event_id,
                    customer_id=cust.customer_id,
                    kind=intended,
                    is_contact=intended.is_contact,
                    amount_paise=ev.amount_paise,
                    discount_bps=1500 if intended is ActionKind.NUDGE_WITH_INCENTIVE else 0,
                    idempotency_key=f"{ev.event_id}:{intended}:{ev.attempt_number}",
                ),
                customer=CustomerState(
                    customer_id=cust.customer_id,
                    has_consent=cust.has_consent,
                    on_dnd_registry=cust.on_dnd_registry,
                    contacts_in_window=tuple(prior),
                    last_contact_at=prior[-1] if prior else None,
                ),
                event=EventState(event_id=ev.event_id, attempts_so_far=ev.attempt_number - 1),
                now=ev.occurred_at,
                config=cfg,
            )
        )

        # A vetoed action does not silently become a different action. It becomes
        # NO_ACTION, and the reason is recorded. Substituting a "safer" message here
        # is exactly the bug that makes compliance layers decorative.
        #
        # NEEDS_APPROVAL is a third state, not a denial. It parks the action in a
        # human queue. Folding it into NO_ACTION -- which this code did until the
        # decision inspector surfaced it -- silently drops precisely the
        # highest-value events, with nothing in the metrics to show they existed.
        if verdict.allowed:
            executed = intended
        elif verdict.disposition is Disposition.NEEDS_APPROVAL:
            executed = ActionKind.QUEUED_FOR_APPROVAL
        else:
            executed = ActionKind.NO_ACTION
        if executed.is_contact:
            prior.append(ev.occurred_at)

        recovered = outcome_rng.random() < gt.probability(executed)
        outcomes.append(
            Outcome(
                event_id=ev.event_id,
                stratum=strata[ev.event_id],
                arm=arm,
                intended=intended,
                executed=executed,
                gate_denials=tuple(str(r.gate) for r in verdict.denials),
                needed_approval=bool(verdict.approvals_needed),
                recovered=recovered,
                amount_paise=ev.amount_paise,
                was_contact=executed.is_contact,
            )
        )

    return Report(
        strategy=name,
        outcomes=tuple(outcomes),
        scenario_total_paise=scenario.total_at_risk_paise,
        ceiling_paise=scenario.ceiling_paise(),
        self_heal_paise=scenario.self_heal_paise(),
    )


def compare(reports: list[Report]) -> str:
    """The table that goes in the pitch video."""
    w = 118
    lines = [
        "=" * w,
        f"{'strategy':<18}{'GROSS':>13}{'INCREMENTAL':>14}{'lift':>8}"
        f"{'95% CI':>18}{'contacts':>10}{'blocked':>9}{'Rs/contact':>12}{'capture':>9}",
        "-" * w,
    ]
    for r in reports:
        lo, hi = r.lift_ci()
        lines.append(
            f"{r.strategy:<18}"
            f"{r.gross_paise / 100:>13,.0f}"
            f"{r.incremental_paise / 100:>14,.0f}"
            f"{r.lift:>7.2%} "
            f"{f'[{lo:+.2%},{hi:+.2%}]':>18}"
            f"{r.contacts:>10,}"
            f"{r.blocked:>9,}"
            f"{r.paise_per_contact / 100:>12,.0f}"
            f"{r.capture_of_contestable:>8.1%}"
        )
    lines.append("=" * w)
    if reports:
        r = reports[0]
        lines += [
            f"at risk      Rs.{r.scenario_total_paise / 100:>13,.0f}",
            f"self-heal    Rs.{r.self_heal_paise / 100:>13,.0f}   arrives with no intervention",
            f"ceiling      Rs.{r.ceiling_paise / 100:>13,.0f}   "
            "best possible action on every event",
            f"contestable  Rs.{(r.ceiling_paise - r.self_heal_paise) / 100:>13,.0f}   "
            "the only money any strategy can actually win",
        ]
    return "\n".join(lines)


def mean_lift(reports: list[Report]) -> float:
    return statistics.fmean(r.lift for r in reports) if reports else 0.0
