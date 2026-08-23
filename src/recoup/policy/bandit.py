"""Contextual bandit over the actions the taxonomy says are possible.

The layering matters more than the algorithm:

    taxonomy    decides which actions are ADMISSIBLE   -- a question of fact
    bandit      decides which admissible action to TAKE -- a question of evidence
    gates       decide whether it may actually run      -- a question of permission

Exploration happens strictly inside the admissible set. A bandit allowed to explore
freely would eventually try retrying an expired card a few thousand times to learn
what the error taxonomy already states as fact, and would try messaging people whose
failure had nothing to do with them. That is not exploration, it is rediscovering
documented behaviour at customers' expense. Anything knowable without an experiment
should not be learned by experiment.

Three departures from a textbook LinUCB, and the first was learned the hard way.

**Statically impossible actions are filtered, and vetoed ones still update the
covariance.** The first version of this bandit proposed `nudge_with_incentive` on
3,052 of 5,000 events and got 2 contacts out the far end. The incentive carries a
15% discount, standing authority is 10%, so every single one was parked for
approval. It never executed, so `learn()` never fired, so its arm never accumulated
evidence -- and in LinUCB an arm with no evidence keeps maximal uncertainty and
therefore maximal exploration bonus. Forever. The arm that could never run was
permanently the most attractive one.

That is starvation-by-veto, and it generalises past this one discount: any action a
gate systematically blocks becomes a black hole the bandit falls into. The fix has
two halves.

  1. Constraints knowable at selection time -- a discount exceeding standing
     authority -- are filtered out of the admissible set. The bandit should not be
     choosing between options one of which is already impossible.
  2. Contextual vetoes that are NOT knowable up front (quiet hours, consent,
     fatigue) call `register_blocked`, which updates the arm's covariance matrix
     but NOT its reward vector. Uncertainty falls because we have now seen this
     context; the reward estimate does not move because nothing happened. Teaching
     the bandit that "messaging at 2am does not work" would be a lie -- we never
     messaged anyone.

Two further departures, both because this bandit spends real money:

**Exploration is value-aware.** Standard bandits explore uniformly with respect to
stake, which is indefensible when one arm-pull is a Rs.99 subscription and the next
is a Rs.90,000 invoice. Above a configurable amount the bandit stops exploring and
plays its best current estimate. The information from one expensive event is worth
no more than from one cheap one -- the regret is worth 900x more. So we buy
information where it is cheap and exploit where it is not.

**The reward is not recovery, it is recovery ABOVE DOING NOTHING.** This is the
same mistake the whole project exists to avoid, and it bit here too. A bandit
rewarded on raw recovery learns that NO_ACTION is excellent on exactly the events
where it is worthless: a GATEWAY_DOWN failure self-heals 52% of the time, so
doing nothing scores 0.52 and looks like a triumph. It is chasing gross, not lift.

So non-null arms are rewarded on their ADVANTAGE -- the observed outcome minus what
the NO_ACTION arm predicts for the same context -- and the NO_ACTION arm learns the
baseline that makes that subtraction possible. Advantage is zero by construction for
doing nothing, so an action is chosen only when the evidence says it CAUSES
recovery, not merely that recovery follows it.

Measured on 5,000 events, seed 20260905: raw-reward 7.45% lift, advantage-reward
9.30%. Across four seeds the bandit averages 9.15% (sd 2.34%) on 436 contacts.

A contact penalty is subtracted on top, because a policy optimising pure
incremental recovery still converges on messaging everybody. The penalty encodes
what the contact budget encodes elsewhere: attention is a finite resource that
belongs to the customer, not to us.

Where this lands, honestly
--------------------------
The deterministic `taxonomy_policy` still wins on lift: 10.67% (sd 2.15%) against
the bandit's 9.15% (sd 2.34%) across four seeds. The intervals overlap heavily, so
the gap is not resolved at this sample size -- but the bandit does not beat it, and
saying otherwise would be the easiest lie in this repository to tell.

That result makes sense rather than being a failure to explain away. The structure
of this problem is genuinely KNOWN: an expired card is not recoverable by retry, and
the taxonomy states it as fact. A policy that encodes known structure should beat one
that must rediscover it from noisy outcomes, and if it did not, the taxonomy would
be wrong. The bandit spends real regret learning things the table already contains.

Where a bandit should earn its place is precisely where the table is SILENT -- which
of several admissible contact variants to send, at what hour, on which channel, to
which segment. Those are questions of evidence with no documented answer, and they
are the honest next step for this component. Deploying it today would trade 1.5
points of lift for a learning capability aimed at questions this version does not
yet ask, so `taxonomy_policy` remains the default and the bandit ships behind a flag.

The prior does not currently help: 8.86% with it against 9.15% without. The offline
model is trained on a uniform-random action log, so its NO_ACTION estimate -- the
baseline every advantage is measured against -- is the weakest part of it, and a
biased baseline biases every other arm. Reported rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recoup.diagnosis.taxonomy import Diagnosis, RetryClass
from recoup.domain import ActionKind, AtRiskEvent, Customer, RiskKind
from recoup.policy.gates import PolicyConfig
from recoup.policy.propensity import PropensityModel, featurise

INCENTIVE_BPS = 1500
"""Discount carried by NUDGE_WITH_INCENTIVE. Must be checked against standing
authority before the action is offered to the bandit at all -- see the module
docstring on starvation-by-veto."""

CONTACT_PENALTY = 0.15
"""Reward subtracted for touching a customer.

Roughly: a contact must lift recovery probability by 15 points to be worth making.
Tuned to the point where the bandit stops choosing contact for causes the taxonomy
marks as not-customer-actionable even before admissibility filters them out --
a useful sanity property, since the two mechanisms should agree.
"""

EXPLORE_CEILING_PAISE = 10_000_00
"""Above Rs.10,000, exploit rather than explore. See the module docstring."""

ALPHA = 0.6
"""UCB width. Higher explores more; 0 is pure greedy."""


def admissible(
    event: AtRiskEvent,
    diagnosis: Diagnosis | None,
    config: PolicyConfig | None = None,
) -> list[ActionKind]:
    """Which actions are even possible for this failure.

    Pure function of the taxonomy plus standing authority. No learning, no
    probabilities -- these are statements of fact about the failure and about what
    this system is permitted to do unattended, and a bandit has no business
    relitigating either.
    """
    cfg = config if config is not None else PolicyConfig()
    # An incentive nobody may authorise unattended is not an option, it is a trap:
    # the bandit would keep choosing it, keep having it parked, and keep learning
    # nothing. See the module docstring.
    incentive_ok = cfg.max_discount_bps >= INCENTIVE_BPS

    if event.kind is RiskKind.CHECKOUT_ABANDONED:
        options = [ActionKind.NO_ACTION, ActionKind.NUDGE]
        if incentive_ok:
            options.append(ActionKind.NUDGE_WITH_INCENTIVE)
        return options
    if event.kind is RiskKind.INVOICE_OVERDUE:
        options = [ActionKind.NO_ACTION, ActionKind.NUDGE, ActionKind.RETRY_SCHEDULED]
        if incentive_ok:
            options.append(ActionKind.NUDGE_WITH_INCENTIVE)
        return options

    # Unrecognised, or not recoverable revenue at all. One option, and it is a human.
    if diagnosis is None or not diagnosis.in_scope:
        return [ActionKind.ROUTE_TO_OPS]

    options = [ActionKind.NO_ACTION]

    if not diagnosis.new_instrument and diagnosis.retry_class is not RetryClass.NEVER:
        # Retrying the same instrument is not provably futile, so it is on the table.
        options.append(
            ActionKind.RETRY_NOW
            if diagnosis.retry_class is RetryClass.NOW
            else ActionKind.RETRY_SCHEDULED
        )

    if diagnosis.customer_action:
        # A human has to do something, so telling them is at least coherent.
        options.append(
            ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH
            if diagnosis.new_instrument
            else ActionKind.NUDGE
        )
        if incentive_ok:
            options.append(ActionKind.NUDGE_WITH_INCENTIVE)
    elif not diagnosis.retryable:
        # Nothing to retry and nobody to tell.
        options.append(ActionKind.ROUTE_TO_OPS)

    return options


def reward(recovered: bool, action: ActionKind, penalty: float = CONTACT_PENALTY) -> float:
    """What the bandit is actually maximising."""
    return (1.0 if recovered else 0.0) - (penalty if action.is_contact else 0.0)


class _Arm:
    """Ridge regression state for one action. LinUCB, per arm.

    `a` starts at the identity, which is the ridge prior: with no evidence the arm
    has a wide confidence interval, which is exactly what makes an unexplored action
    attractive. That property is also what made a permanently-vetoed action a black
    hole until `register_blocked` existed.
    """

    __slots__ = ("a", "b", "blocked", "dim", "pulls")

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.a = np.identity(dim)
        self.b = np.zeros(dim)
        self.pulls = 0
        self.blocked = 0

    def theta(self) -> np.ndarray:
        return np.asarray(np.linalg.solve(self.a, self.b))

    def score(self, x: np.ndarray, alpha: float) -> tuple[float, float]:
        """Return (point estimate, uncertainty bonus)."""
        a_inv_x = np.linalg.solve(self.a, x)
        mean = float(self.theta() @ x)
        variance = max(0.0, float(x @ a_inv_x))
        return mean, alpha * float(np.sqrt(variance))

    def update(self, x: np.ndarray, r: float) -> None:
        self.a += np.outer(x, x)
        self.b += r * x
        self.pulls += 1


@dataclass(frozen=True, slots=True)
class Choice:
    """What the bandit picked, and enough of why to write to the ledger."""

    action: ActionKind
    expected_reward: float
    uncertainty: float
    explored: bool
    admissible: tuple[ActionKind, ...]

    def explain(self) -> str:
        mode = "explored" if self.explored else "exploited"
        return (
            f"{self.action} ({mode}, E[r]={self.expected_reward:+.3f} "
            f"±{self.uncertainty:.3f}, from {len(self.admissible)} admissible)"
        )


class RecoveryBandit:
    """LinUCB over admissible actions, with a value-aware exploration ceiling."""

    def __init__(
        self,
        dim: int,
        *,
        alpha: float = ALPHA,
        explore_ceiling_paise: int = EXPLORE_CEILING_PAISE,
        contact_penalty: float = CONTACT_PENALTY,
        prior: PropensityModel | None = None,
        config: PolicyConfig | None = None,
    ) -> None:
        self.dim = dim
        self.config = config if config is not None else PolicyConfig()
        self.alpha = alpha
        self.explore_ceiling_paise = explore_ceiling_paise
        self.contact_penalty = contact_penalty
        # The propensity model is a warm start, not a replacement. A cold bandit
        # spends its first few thousand events learning things the model already
        # knows, and those events are real customers.
        self.prior = prior
        self._arms: dict[ActionKind, _Arm] = {}

    def _arm(self, action: ActionKind) -> _Arm:
        if action not in self._arms:
            self._arms[action] = _Arm(self.dim)
        return self._arms[action]

    def _baseline(
        self, event: AtRiskEvent, diagnosis: Diagnosis | None, customer: Customer
    ) -> float:
        """What we expect to recover on this event if we do nothing at all.

        The counterfactual every other arm is measured against. Estimated from the
        NO_ACTION arm, which is the one arm that learns raw outcomes rather than
        advantages -- and it is never starved of data, because doing nothing is
        always admissible.
        """
        arm = self._arm(ActionKind.NO_ACTION)
        x0 = featurise(event, diagnosis, customer, ActionKind.NO_ACTION)
        if arm.pulls < 20 and self.prior is not None:
            return self.prior.predict(event, diagnosis, customer, ActionKind.NO_ACTION)
        return float(arm.theta() @ x0)

    def select(
        self,
        event: AtRiskEvent,
        diagnosis: Diagnosis | None,
        customer: Customer,
    ) -> Choice:
        options = admissible(event, diagnosis, self.config)
        if len(options) == 1:
            return Choice(options[0], 0.0, 0.0, False, tuple(options))

        # Above the ceiling, uncertainty is not worth paying for.
        exploring = event.amount_paise <= self.explore_ceiling_paise
        alpha = self.alpha if exploring else 0.0

        best: tuple[float, ActionKind, float, float] | None = None
        for action in options:
            x = featurise(event, diagnosis, customer, action)
            mean, bonus = self._arm(action).score(x, alpha)

            if action is ActionKind.NO_ACTION:
                # Advantage over itself is zero by definition. Every other arm has to
                # beat this to be chosen, which is the whole point.
                mean = 0.0
            elif self.prior is not None and self._arm(action).pulls < 30:
                # Blend toward the offline model while this arm is still ignorant.
                # Weight decays as evidence arrives, so the prior informs the cold
                # start and then gets out of the way. Expressed as an advantage, to
                # match what the arm itself is learning.
                p = self.prior.predict(event, diagnosis, customer, action)
                p0 = self.prior.predict(event, diagnosis, customer, ActionKind.NO_ACTION)
                prior_adv = (p - p0) - (self.contact_penalty if action.is_contact else 0.0)
                w = 1.0 - self._arm(action).pulls / 30.0
                mean = (1 - w) * mean + w * prior_adv

            total = mean + bonus
            if best is None or total > best[0]:
                best = (total, action, mean, bonus)

        assert best is not None
        _, action, mean, bonus = best
        return Choice(
            action=action,
            expected_reward=mean,
            uncertainty=bonus,
            explored=exploring and bonus > 0.0,
            admissible=tuple(options),
        )

    def update(
        self,
        event: AtRiskEvent,
        diagnosis: Diagnosis | None,
        customer: Customer,
        action: ActionKind,
        recovered: bool,
    ) -> None:
        x = featurise(event, diagnosis, customer, action)
        outcome = 1.0 if recovered else 0.0
        if action is ActionKind.NO_ACTION:
            # The baseline arm learns the raw outcome; everything else is measured
            # against it.
            self._arm(action).update(x, outcome)
            return
        advantage = outcome - self._baseline(event, diagnosis, customer)
        penalty = self.contact_penalty if action.is_contact else 0.0
        self._arm(action).update(x, advantage - penalty)

    def register_blocked(
        self,
        event: AtRiskEvent,
        diagnosis: Diagnosis | None,
        customer: Customer,
        action: ActionKind,
    ) -> None:
        """A gate vetoed this action. Record the context, but not a reward.

        Updates the arm covariance (A) and deliberately NOT the reward vector (b).
        Uncertainty falls because we have now seen this context; the reward estimate
        does not move, because nothing happened and we learned nothing about what
        would have. Without this an action a gate blocks systematically keeps maximal
        uncertainty and therefore maximal exploration bonus forever, and the bandit
        proposes it on every event for the rest of time.
        """
        x = featurise(event, diagnosis, customer, action)
        arm = self._arm(action)
        arm.a += np.outer(x, x)
        arm.blocked += 1

    @property
    def pulls(self) -> dict[str, int]:
        return {str(a): arm.pulls for a, arm in sorted(self._arms.items())}

    @property
    def blocked(self) -> dict[str, int]:
        return {str(a): arm.blocked for a, arm in sorted(self._arms.items())}
