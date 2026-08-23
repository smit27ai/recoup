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
| Unmapped / new codes | `claude-opus-5`, strict tool schema | Genuine ambiguity — and it proposes a table row for human review |
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

One command, no configuration, no credentials, no network:

```bash
python -m recoup -n 300
```

It runs the full path -- diagnose, decide, gate, execute, record -- over 300
synthetic at-risk events and prints what happened to every one of them, plus a
handful of decisions in full. With `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` set it
drives real test-mode Razorpay instead; same engine, same gates, same ledger, only
the transport changes.

```
  WHY ACTIONS WERE BLOCKED
  ------------------------------------------------------------------
  quiet_hours                                     103
  consent                                          29
  dnd                                              20
  fatigue                                           5

  value we chose NOT to chase   Rs.     1,021,764
  (compliance is a cost. Not measuring it is how it quietly stops being real.)

  # held out, so we did nothing on purpose
  [5] 2026-08-05T18:56:00+05:30  event=evt_000005 customer=cust_00080  Rs.197.00
    saw       transaction_limit_exceeded -> LIMIT_EXCEEDED (tier 1)
    wanted    nudge_with_instrument_switch
      ok   quiet_hours          18:56 inside 08:00-19:00
      ok   stopping_rule        attempt 2/4
    did       no_action   [allow]
    outcome   open   arm=holdout

  INTEGRITY
  ------------------------------------------------------------------
  ledger              300 records, hash chain verified
  messages sent       32
  holdout contacted   0  (must be 0, or measurement is void)
  approval queue      14 items, Rs.292,168
```

Note the holdout trace: every gate said **allow**. The holdout is what stopped it.
Those are different reasons for doing nothing and the ledger distinguishes them,
which is what makes the control arm interpretable rather than a hole in the data.

Setup:

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
```

```bash
PYTHONPATH=src .venv/Scripts/python -m pytest -q
```

```bash
PYTHONPATH=src .venv/Scripts/python -c "from recoup.generator.synthetic import ScenarioGenerator; from recoup.measure.harness import run, compare; from recoup.policy.strategies import STRATEGIES; s=ScenarioGenerator().generate(5000); print(compare([run(s,f,n) for n,f in STRATEGIES.items()]))"
```

## Audit trail

Every decision lands in an append-only, hash-chained ledger answering six questions:
what we saw, what we wanted, what we were allowed, what we did, what happened, and
which arm it was in. Gate reasons are stored verbatim, not re-derived. Editing,
deleting, reordering, or re-sealing a record all fail verification.

```
[2] 2026-09-01T03:15:00+05:30  event=evt_0002 customer=cust_0002  Rs.12,999.00
  saw       gateway_technical_error -> GATEWAY_DOWN (tier 1)
  wanted    retry_now
    ok   consent              not a contact action
    ok   quiet_hours          not a contact action
    HOLD value_approval       Rs.12,999 exceeds unattended threshold Rs.5,000
    ok   idempotency          key unused
  did       queued_for_approval   [needs_approval]
  outcome   open   arm=treatment
```

Rendering one decision as readable text immediately exposed a bug that every
aggregate metric had hidden: `needs_approval` was collapsing into `no_action`,
silently dropping **536 events worth Rs.60,28,045 — 21% of all at-risk money** —
with no queue entry and nothing in the metrics. Aggregates hide state-machine bugs
by construction.

## Status

Day 1 of 13. The execution path runs end to end: webhook -> diagnose (tier 1, then
tier 2) -> policy -> gates -> Razorpay -> ledger, with uncertain-outcome
reconciliation and a verifiable audit trail. **179 tests, ruff clean, mypy --strict
clean.**

Not yet built: propensity model, bandit, durable multi-day workflows.

## Ops console

Two queues in this system can only be drained by a human: **approvals** (actions over
the unattended authority limit) and **rules** (tier-2 proposals awaiting promotion).
A queue nobody can drain is not a safety mechanism, it is a place money goes to die
— Recoup parks ~21% of at-risk value in the approval queue *by design*, and without
a way to work it that design is just a slower kind of losing.

```bash
python -m recoup.console.server     # API + seeded data on :8000
npm --prefix console run dev        # console on :5173
```

Four views: an overview that puts **the cost of compliance in rupees** next to what
was recovered; a decision list with a drawer showing every gate that ran in the words
it used at the time; the approval queue; and the rule-review queue.

Two properties it is built around:

- **Reviews append, never edit.** Approving a parked action writes a *new* ledger
  record naming the reviewer. The original said "no human has looked yet" and that
  stays true of the moment it describes. A console that could rewrite history would
  destroy the exact property it exists to expose — and the header shows live chain
  verification, reporting a break rather than hiding it.
- **Promoting a rule is validated before it is written.** A rule that would not parse
  is rejected rather than appended, because a taxonomy that fails to load takes the
  system down at the next restart, long after the reviewer who broke it went home.

Approving a rule closes the loop end to end: the code moves from tier 2 to tier 1,
is resolved by table lookup from then on with no model call, and contact unlocks
through the ordinary path — having been seen by a person.

## Tier 2: where a model earns its place

Tier 1 resolves ~110 documented error reasons by table lookup. Tier 2 exists only
for what tier 1 returns `None` on — a code Razorpay adds after we ship. That is a
genuinely ambiguous language problem, which is what a model is good at, unlike
looking things up in a table, which it is bad at.

**Trust is graded by blast radius, not by the model's confidence.**

| consequence | customer impact | authority required |
|---|---|---|
| route to ops | none | any confidence — being wrong costs an ops ticket |
| silent retry | near-zero | enough to be worth one API call (0.5) |
| **contact a customer** | high, irreversible | **human review. No confidence value suffices.** |

Tier 2 may route and may retry. It may **never** authorise contacting a customer —
not at 0.99, not ever. The gate on contact is a human approving the mined rule,
because a model can be confidently wrong and its confidence is precisely the thing
you cannot check at 3am. A parametrised test sweeps the whole confidence range to
make sure nobody reintroduces a threshold there.

```
unmapped code               tier1   tier2 root cause        retry?  contact?
----------------------------------------------------------------------------
card_has_expired_2027       MISS    INSTRUMENT_INVALID      no      no
acct_balance_shortfall      MISS    FUNDS                   yes     no
merchant_kyc_pending        MISS    MERCHANT_CONFIG         no      no
issuer_host_unreachable     MISS    ISSUER_DOWN             yes     no
zx_qq_9917                  MISS    UNKNOWN                 no      no
```

**Every escalation mines a rule.** Each answer is also a candidate row for
`error_taxonomy.tsv`, ranked by how often the code was seen. Approve one and that
code is tier 1 forever after — free, instant, deterministic, and now permitted to
drive contact through the ordinary path. The model's job is to shrink its own job.

**One unknown code costs one call, not N.** Cached by reason. That is a consistency
argument before it is a cost argument: two identical failures must never get
different diagnoses because the sampler went a different way. Failures are cached
too, so a model outage is asked once rather than once per event.

**Tier 2 is an enhancement, never a dependency.** If the API is down, rate-limited,
or returns nonsense, the code routes to a human — exactly what happens with no tier
2 configured at all. Without `ANTHROPIC_API_KEY` a deterministic offline stub takes
over, and the `anthropic` SDK is an optional extra (`pip install -e ".[llm]"`).

### How the layers are separated

`decide()` is pure and authorises nothing. `authorise()` runs the gates against the
clock **at the moment of action**, never cached from decision time -- a workflow
that slept four days may have been planned inside quiet hours and woken outside
them, or the customer may have revoked consent while it slept. `execute()` is the
only code permitted a side effect, and it never re-decides: if a payment link
cannot be raised, that is a failure to record, not licence to send an SMS instead.

The holdout is enforced at the top of the pipeline rather than in policy, so a
control-arm event still goes through diagnosis and policy and the ledger records
what we *would* have done.

## Handling money safely

Razorpay provides **no server-side idempotency** on the endpoints a recovery
workflow uses — `X-Payout-Idempotency` covers only Payouts, Composite APIs, and the
idempotent Refund/Route variants. Orders, Payment Links and Subscription charges
have none, so a retried POST creates a *second* order. In a recovery system that is
a double charge against someone who already paid.

Consequently:

- **A timeout is an unknown, not a failure.** Mutating calls that end ambiguously
  raise `UncertainOutcome`; the caller must reconcile by looking the entity up via a
  receipt chosen *before* the call. Blind retry double-charges; assuming failure
  loses money silently. `receipt` is therefore required, not optional.
- **5xx and 429 are retried on POST** (Razorpay is telling us it never processed the
  request). **408 is not** — a gateway timeout carries the same ambiguity as a
  client-side one.
- **Live keys are refused** unless `allow_live=True`. This project has no business
  being one environment variable away from moving real money.
- **Razorpay's own SMS/email on payment links is disabled.** Letting the processor
  notify the customer would route around consent, DND, quiet hours and the contact
  budget. Contact goes through the gated path or not at all.

## What this deliberately does not do

Real WhatsApp/BSP delivery, real money movement, voice recovery, multi-tenant auth.
Thirteen days buys depth on one thing or a demo of six. The track bar asks for
measured money with an audit trail, so the depth went there.
