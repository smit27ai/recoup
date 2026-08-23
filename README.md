# Recoup

**A recovery decisioning control plane.** Every at-risk rupee gets a diagnosis, a
bounded intervention chosen under hard compliance gates, and — the part almost
nobody does — a randomised holdout that proves the money was *incremental*.

Razorpay AI Buildathon 2026 · Track 3, AI Revenue Recovery

---

## The problem it actually solves

Razorpay [Agent Studio](https://razorpay.com/agent-studio/) already ships Subscription
Recovery, Abandoned Cart Conversion, Dispute Responder, RTO Shield, Settlement
Insights and Cashflow Forecaster. Nearly every example direction in the track brief
is a product they already run in production.

What seven single-purpose recovery agents do not have is a layer above them. Nothing
arbitrates when three of them want to message the same customer on the same morning.
Nothing budgets contact across them. Nothing enforces a stopping rule when the
customer has already promised to pay. And nothing establishes that the money they
report actually arrived *because* of them.

Recoup is that layer.

## The one number that matters

Every dunning product reports **gross** recovery — every rupee that arrived on an
account it touched. A large share of that money arrives on its own: the issuer comes
back up, the customer retries, the card clears on payday. Counting it as recovery is
free credit for work nobody did.

Recoup randomises eligible events into treatment and a holdout, and reports the
difference.

```
strategy              GROSS   INCREMENTAL    lift            95% CI  contacts  Rs/contact
no_action         3,784,720      -363,084  -1.27%   [-4.00%,+1.61%]         0           0
blind_retry       4,074,026     1,254,403   4.38%   [+1.69%,+7.46%]         0           0
blast             3,981,997       767,725   2.68%   [+0.02%,+5.46%]       827         928
taxonomy_policy   4,371,053     2,950,618  10.30%  [+7.47%,+13.18%]       595       4,959

at risk      Rs. 28,649,566
self-heal    Rs.  5,060,380   arrives with no intervention
ceiling      Rs. 14,065,687   best possible action on every event
contestable  Rs.  9,005,306   the only money any strategy can actually win
```

Gross recovery varies **15%** across these strategies. Incremental recovery varies
**8×**. Judged on gross, blind retry looks like a fine product. Judged on what it
caused, it captures 13.9% of the winnable money while the policy captures 32.8%.

`taxonomy_policy` uses **28% fewer contacts** than blast and recovers **3.8× more**
real money — because knowing that an expired card can never be recovered by a retry
is worth more than any amount of retry budget.

## Where AI is used, and where it deliberately is not

| Component | Tool | Why |
|---|---|---|
| Root cause from error code | **Lookup table**, 110 reasons | Deterministic, auditable, 0ms, free |
| Unmapped / new codes | LLM escalation *(day 4)* | Genuine ambiguity — and it proposes a table row for human review |
| Recovery propensity | **Gradient boosting** *(day 5)* | The answer is a number, not prose |
| Intervention choice | **Contextual bandit** *(day 6)* | Explore/exploit is arithmetic |
| Compliance gates | **Pure predicates** | Must be provable. Never probabilistic |
| Message copy | LLM *(day 7)* | High volume, genuinely generative |

A language model is very bad at looking things up in a table, and a table is very bad
at writing Hinglish. The split follows from that.

## Architecture

```
Razorpay test-mode APIs + webhooks ──┐
Synthetic generator (ground truth)  ─┴──▶ Event Ledger (append-only)
                                              │
                                    ┌─────────▼─────────┐
                                    │ 1. DIAGNOSIS      │ 110 reasons -> 18 root
                                    │   table, then LLM │ causes; 32 route to ops
                                    └─────────┬─────────┘
                                    ┌─────────▼─────────┐
                                    │ 2. POLICY         │ deterministic routing,
                                    │   pick an action  │ bandit on top
                                    └─────────┬─────────┘
                                    ┌─────────▼─────────┐
                                    │ 3. GATES  (VETO)  │ consent · DND · quiet hours
                                    │   evaluated at    │ contact budget · fatigue
                                    │   EXECUTION time  │ stopping rules · authority
                                    └─────────┬─────────┘
                                    ┌─────────▼─────────┐
                                    │ 4. EXECUTOR       │ idempotent, multi-day,
                                    │                   │ human approval over limits
                                    └─────────┬─────────┘
                    ┌──────────────────────────┴──────────────────────────┐
          ┌─────────▼─────────┐                              ┌────────────▼──────┐
          │ 5. INCREMENTALITY │                              │ 6. AUDIT LEDGER   │
          │  holdout, lift+CI │                              │  hash-chained     │
          └───────────────────┘                              └───────────────────┘
```

### Two design rules that carry most of the weight

**Gates never short-circuit.** All nine run on every decision. An action blocked for
three independent reasons must be auditable as blocked for three reasons. Stopping at
the first denial saves microseconds and destroys the audit trail.

**Gates run immediately before execution, never at planning time.** A workflow that
slept four days may have been planned inside quiet hours and woken outside them, or
the customer may have revoked consent while it slept. Deciding early and executing
late is the standard way a compliant system emits non-compliant messages.

**A vetoed action becomes `NO_ACTION`, never a "safer" substitute.** Silently
downgrading a blocked WhatsApp message to an email is how a compliance layer becomes
decorative.

## Compliance

Gate thresholds are operator-configurable policy, each carrying the regime it derives
from: RBI Fair Practices Code (recovery contact hours), TRAI TCCCPA/DLT (commercial
messaging preference registries), DPDP Act 2023 (lawful basis for contact). These are
defaults reflecting our reading, not legal advice — see `src/recoup/policy/gates.py`.
Verify with counsel before any production use.

## Measurement, and why it is not the obvious design

The measurement rig failed its own A/A test three times on day 1, once producing a
**confident false positive** — a −4.44% reading with a CI excluding zero, for a
strategy that does nothing. The estimator and stratification design are the result of
a measured sweep, not intuition, and both findings were counter to expectation:

- The pooled count-rate estimator beat every stratified/value-weighted variant on
  bias *and* variance.
- Stratifying assignment on **amount** was *worse than not stratifying at all*;
  stratifying on **root cause × amount** was best. Stratify on what predicts the
  outcome first, what scales the value second.

Full account in [`JOURNAL.md`](JOURNAL.md). Both findings are locked in by tests —
if `test_aa_null` fails, no other number this project reports can be trusted.

## Running it

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
```

```bash
PYTHONPATH=src .venv/Scripts/python -m pytest -q
```

```bash
PYTHONPATH=src .venv/Scripts/python -c "from recoup.generator.synthetic import ScenarioGenerator; from recoup.measure.harness import run, compare; from recoup.policy.strategies import STRATEGIES; s=ScenarioGenerator().generate(5000); print(compare([run(s,f,n) for n,f in STRATEGIES.items()]))"
```

## Status

Day 1 of 13. Built: failure taxonomy (110 reasons), compliance gate layer (9 gates),
ground-truth simulator, measurement harness, 40 tests.

Not yet built: LLM escalation tier, propensity model, bandit, Razorpay test-mode
execution path, durable workflows, ops console.

## What this deliberately does not do

Real WhatsApp/BSP delivery, real money movement, voice recovery, multi-tenant auth.
Thirteen days buys depth on one thing or a demo of six. The track bar asks for
measured money with an audit trail, so the depth went there.
