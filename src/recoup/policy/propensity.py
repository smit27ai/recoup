"""Recovery propensity: P(this event is recovered | we take this action).

The bandit downstream consumes these numbers as expected values and multiplies them
by rupees, so the property that matters here is **calibration, not accuracy**. A
model that ranks events perfectly but says 0.9 when the truth is 0.4 will make the
bandit systematically over-act -- it will contact people it should have left alone,
and it will look like it is working right up until someone measures incrementality.
Ranking metrics (AUC) cannot see that failure at all. `reliability()` and
`expected_calibration_error()` exist because those are the numbers worth reporting.

Two design decisions worth stating.

**Trained on logged decisions, not on ground truth.** The simulator knows every
event's true recovery probability under every action, and the model is never shown
any of it. Instead it learns from a log of (context, action, outcome) triples
produced by an exploring policy -- exactly the data a real deployment has. Training
on the answer key would produce a model that cannot exist in production and numbers
nobody should believe.

**The action is a feature, not a separate model.** One model over (context, action)
rather than one model per action: the actions share almost all their structure
(an expired card is unrecoverable by retry regardless of which retry), so splitting
them throws away data and leaves the rare actions badly estimated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from recoup.diagnosis.taxonomy import Diagnosis, RetryClass, RootCause, diagnose
from recoup.domain import ActionKind, AtRiskEvent, Channel, Customer, RiskKind

if TYPE_CHECKING:
    from recoup.generator.synthetic import Scenario

ROOT_CAUSES: list[object] = list(RootCause)
RETRY_CLASSES: list[object] = list(RetryClass)
ACTIONS: list[object] = list(ActionKind)
SEGMENTS: list[object] = ["new", "casual", "loyal", "at_risk", "business"]
CHANNELS: list[object] = list(Channel)
RISK_KINDS: list[object] = list(RiskKind)


def _onehot(value: object, options: list[object]) -> list[float]:
    return [1.0 if value == o else 0.0 for o in options]


def featurise(
    event: AtRiskEvent,
    diagnosis: Diagnosis | None,
    customer: Customer,
    action: ActionKind,
) -> np.ndarray:
    """Context and action as one vector.

    Deliberately excludes anything a real deployment would not have at decision
    time. No outcome, no ground truth, and nothing derived from the future -- a
    feature that leaks is how a model gets 0.99 offline and fails in production.
    """
    return np.array(
        [
            *_onehot(diagnosis.root_cause if diagnosis else None, ROOT_CAUSES),
            *_onehot(diagnosis.retry_class if diagnosis else None, RETRY_CLASSES),
            1.0 if diagnosis and diagnosis.new_instrument else 0.0,
            1.0 if diagnosis and diagnosis.customer_action else 0.0,
            1.0 if diagnosis and diagnosis.in_scope else 0.0,
            1.0 if diagnosis is None else 0.0,  # unmapped code is itself a signal
            # Log scale: the difference between Rs.99 and Rs.999 matters far more
            # than between Rs.90,000 and Rs.90,900.
            float(np.log1p(event.amount_paise / 100.0)),
            float(event.attempt_number),
            *_onehot(customer.segment, SEGMENTS),
            *_onehot(customer.preferred_channel, CHANNELS),
            *_onehot(event.kind, RISK_KINDS),
            *_onehot(action, ACTIONS),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class LoggedDecision:
    """One row of training data, exactly as production would record it."""

    features: np.ndarray
    recovered: bool


@dataclass(frozen=True, slots=True)
class Reliability:
    """How well the predicted probabilities match observed frequencies."""

    bins: tuple[tuple[float, float, int], ...]
    """(mean predicted, observed frequency, count) per bin."""
    ece: float
    """Expected calibration error: average |predicted - observed|, count-weighted."""
    brier: float

    def report(self) -> str:
        lines = [
            f"  ECE {self.ece:.4f}   Brier {self.brier:.4f}",
            f"  {'predicted':>12}{'observed':>12}{'n':>8}",
        ]
        for predicted, observed, n in self.bins:
            lines.append(f"  {predicted:>12.3f}{observed:>12.3f}{n:>8}")
        return "\n".join(lines)


class PropensityModel:
    """Calibrated logistic regression over (context, action)."""

    def __init__(self, seed: int = 20260905) -> None:
        self._model: CalibratedClassifierCV | None = None
        self.seed = seed

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, log: list[LoggedDecision]) -> PropensityModel:
        if len(log) < 50:
            raise ValueError(f"need at least 50 logged decisions to fit, got {len(log)}")
        x = np.vstack([row.features for row in log])
        y = np.array([1 if row.recovered else 0 for row in log])
        if len(set(y.tolist())) < 2:
            raise ValueError("training log contains only one outcome class")

        base = LogisticRegression(max_iter=2000, C=1.0, random_state=self.seed)
        # Isotonic calibration on cross-validated folds. Logistic regression is
        # already roughly calibrated on its own; this matters because the features
        # are heavily one-hot and the rarer actions would otherwise inherit the
        # majority action's confidence.
        self._model = CalibratedClassifierCV(base, method="isotonic", cv=5)
        self._model.fit(x, y)
        return self

    def predict(
        self,
        event: AtRiskEvent,
        diagnosis: Diagnosis | None,
        customer: Customer,
        action: ActionKind,
    ) -> float:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        features = featurise(event, diagnosis, customer, action).reshape(1, -1)
        return float(self._model.predict_proba(features)[0, 1])

    def predict_many(self, rows: list[np.ndarray]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(self._model.predict_proba(np.vstack(rows))[:, 1])

    def reliability(self, log: list[LoggedDecision], n_bins: int = 10) -> Reliability:
        """Do the probabilities mean what they say?

        This, not AUC, is the number that decides whether the bandit downstream can
        trust the output as an expected value.
        """
        predicted = self.predict_many([row.features for row in log])
        observed = np.array([1.0 if row.recovered else 0.0 for row in log])

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins: list[tuple[float, float, int]] = []
        total_error = 0.0
        for lo, hi in pairwise(edges):
            mask = (predicted >= lo) & (predicted < hi if hi < 1.0 else predicted <= hi)
            n = int(mask.sum())
            if n == 0:
                continue
            p, o = float(predicted[mask].mean()), float(observed[mask].mean())
            bins.append((p, o, n))
            total_error += n * abs(p - o)

        return Reliability(
            bins=tuple(bins),
            ece=total_error / len(log) if log else 0.0,
            brier=float(np.mean((predicted - observed) ** 2)),
        )


def collect_training_log(
    scenario: Scenario,
    *,
    seed: int = 20260905,
    actions: list[ActionKind] | None = None,
) -> list[LoggedDecision]:
    """Simulate an exploring policy and record what happened.

    This stands in for the logs a real deployment accumulates. The policy is uniform
    random over admissible actions, which is deliberately a bad recovery policy and a
    very good source of training data -- every action gets observed on every kind of
    event, so the model learns what does NOT work as well as what does. A log
    generated by a good policy would never contain the evidence that retrying an
    expired card is futile, because a good policy never does it.
    """
    rng = random.Random(seed)
    pool = actions or [a for a in ActionKind if a is not ActionKind.QUEUED_FOR_APPROVAL]
    log: list[LoggedDecision] = []

    for event in scenario.events:
        customer = scenario.customers[event.customer_id]
        diagnosis = diagnose(event.error_reason)
        action = rng.choice(pool)
        truth = scenario.truth[event.event_id]
        recovered = rng.random() < truth.probability(action)
        log.append(
            LoggedDecision(
                features=featurise(event, diagnosis, customer, action),
                recovered=recovered,
            )
        )
    return log
