# Build journal

Kept daily, committed daily. Incidents as they happened, including the wrong turns,
because the wrong turns are where the actual engineering was.

---

## Day 1 — 2026-08-23

### Decided the track and the angle

Track 3, AI Revenue Recovery. The choice was driven by one observation: Razorpay
already ships [Agent Studio](https://razorpay.com/agent-studio/) with Subscription
Recovery, Abandoned Cart Conversion, Dispute Responder, RTO Shield, Settlement
Insights and Cashflow Forecaster. Nearly every "example direction" in the track brief
maps onto a product they already have in production.

So building an eighth recovery agent means shipping a worse copy of their v1 in
thirteen days. What they have seven of, and none of, is the layer *above*: something
that arbitrates when three of those agents all want to message the same customer on
the same morning, budgets contact, enforces stopping rules, and proves the money was
incremental rather than money that would have arrived anyway.

Re-reading the track bar confirms it — "measured money recovered across a batch, with
compliant escalation, stopping rules, and an audit trail" describes a control plane,
not a bot.

### Grounded the taxonomy in real data instead of memory

I was about to hardcode Razorpay decline codes from recollection. Caught myself: the
panel is Razorpay engineers, and invented error codes would be spotted in seconds.
Scraped the real list from the docs instead (JS-rendered, so `WebFetch` returned an
empty shell and I had to drive a browser to get the table).

110 reasons. The useful move was collapsing them onto the four questions recovery
policy actually asks — root cause, when to retry, whether the same instrument is
futile, whether a human needs to act — which gets 110 codes down to 18 root causes.
32 of the 110 turn out not to be recovery at all; they are our own integration bugs
and ops tickets. Dunning a customer over our own `invalid_order_id` is the most
expensive mistake this system could make, so those route away from the queue entirely.

### The A/A test that broke the measurement rig — three times

This is the day's real work.

Wrote the harness, ran four strategies, and `no_action` reported **+4.88% lift and
₹13.98L recovered**. `no_action` does nothing. Both arms receive `NO_ACTION`. Its
true lift is exactly zero by construction. The rig was inventing money.

**Attempt 1 — diagnose.** Checked the amount distribution: the top 10 events out of
5,000 carry 8.4% of all at-risk value; the top 1% carry 26%. With naive per-event
coin flips, whether four large invoices land in treatment or holdout swings the
value-weighted recovery rate by several points. Real payment portfolios are exactly
this skewed, so this is not a simulator artefact.

**Attempt 2 — stratify assignment by amount decile.** Phantom lift dropped 4.88% →
1.27%. Better, not fixed, and the CIs were still ±7%.

**Attempt 3 — scale up, and get a nastier failure.** At n=60,000 the A/A test
reported **−4.44% with a bootstrap CI that excluded zero**. That is far worse than
the original bug: a confident false positive. The cause was that the bootstrap
resamples *within* the realised assignment, so it measures outcome noise but is blind
to the luck of the split. A single CI cannot tell you your randomisation was unlucky.

**Attempt 4 — post-stratified estimator.** Assumed finer strata plus within-stratum
estimation would fix it. It made things *worse* (A/A sd rose to 4.5%). Splitting into
40 buckets means each bucket's rate comes from ~25 holdout events, and value-weighting
those noisy per-bucket estimates injects more variance than the tail imbalance it
removes.

**What actually worked.** Stopped guessing and ran a sweep: 12 seeded A/A
replications × {pooled, 10, 40, 100, 250 strata} × {count-rate, value-weighted}.

| estimator | A/A mean | sd | detects real policy lift |
|---|---|---|---|
| value-weighted, pooled | +1.87% | 3.81% | +4.31% |
| value-weighted, 40 strata | −1.49% | 4.50% | +0.97% |
| count-rate, 40 strata | −0.14% | 3.86% | +2.33% |
| **count-rate, pooled** | **−0.18%** | **1.24%** | **+10.85%** |

The simple estimator won on both bias *and* variance, and it was the only one that
separated a real strategy from noise. Every "sophistication" I added had been
actively harmful.

**Then a second, subtler bug.** With the estimator fixed, A/A still sat at −1.06%.
The remaining culprit was my stratification key. I had stratified assignment on
**amount**, but amount only scales how much a recovery is *worth* — the variable that
drives *whether it happens* is root cause. Balancing arms on money while leaving the
outcome driver unbalanced turned out to be worse than not stratifying at all:

| assignment strata | A/A mean | sd | max abs err | detects policy lift |
|---|---|---|---|---|
| none | −0.18% | 1.24% | 3.05% | +10.85% |
| amount only | −1.06% | 1.39% | 3.40% | +10.00% |
| root cause only | −0.11% | 1.35% | 1.90% | +10.87% |
| **cause × amount** | **−0.02%** | **1.29%** | **2.36%** | **+11.06%** |

Final: pooled count-rate estimator, assignment stratified on (root cause × amount
quartile). A/A mean −0.02%.

**What I take from it.** Stratify on what predicts the *outcome* first and what
scales the *value* second — I had that backwards for most of the day. And an A/A test
is not a formality: it caught three distinct defects, one of which (the confident
false positive) would have shipped a number I would have defended in a panel and been
wrong about. `test_aa_null` now runs across 12 seeds in CI, and
`test_stratification_beats_amount_only_on_aa` locks in the finding so nobody
"simplifies" it back.

### Where it stands

With a trustworthy rig, the actual result:

```
strategy              GROSS   INCREMENTAL    lift            95% CI  contacts  Rs/contact
no_action         3,784,720      -363,084  -1.27%   [-4.00%,+1.61%]         0           0
blind_retry       4,074,026     1,254,403   4.38%   [+1.69%,+7.46%]         0           0
blast             3,981,997       767,725   2.68%   [+0.02%,+5.46%]       827         928
taxonomy_policy   4,371,053     2,950,618  10.30%  [+7.47%,+13.18%]       595       4,959
```

Gross recovery varies only 15% across strategies. Incremental varies **8×**. That gap
is the entire thesis of the project, and it is now measurable rather than asserted.
`taxonomy_policy` uses 28% *fewer* contacts than `blast` and recovers 3.8× more real
money — because knowing an expired card can never be recovered by a retry is worth
more than any amount of retry budget.

### Audit ledger, and a bug the inspector found immediately

Built the append-only hash-chained ledger (`src/recoup/ledger.py`). Every decision
records six things: what we saw, what we wanted, what we were allowed, what we did,
what happened, and which arm it was in. Gate reasons are stored **verbatim** rather
than as codes to be re-interpreted later against a config that has since changed —
an audit trail that recomputes its own explanations is not evidence.

Tamper tests are the ones that matter: editing a record, editing just a gate's
*reason* string, deleting a record, reordering records, and re-sealing an edited
record after recomputing its hash. All five fail verification. The last one matters
most — partial forgery is not enough, because every later record commits to the old
hash.

Then I printed three sample decisions through `explain()` to check it read well, and
the third one exposed a live bug.

A ₹12,999 gateway retry at 03:15 correctly passed the quiet-hours gate (a silent
retry disturbs nobody — only *contact* is time-restricted) but tripped the value
threshold and came back `needs_approval`. The ledger then showed `did no_action`.

`NEEDS_APPROVAL` is not a denial. It means a human has to look. But the harness had:

```python
executed = intended if verdict.allowed else ActionKind.NO_ACTION
```

...which folds "a human has not looked yet" into "we decided not to act". Those are
different states and must never share a representation. The consequence: **every
event above the approval threshold was silently dropped** — no queue entry, nothing
in the metrics, money simply gone. And it hit precisely the highest-value events,
which are the ones worth most.

Measured after adding `ActionKind.QUEUED_FOR_APPROVAL` and a `queued_paise` metric:
**536 events, ₹60,28,045 — 21% of all at-risk money** was going on the floor.

The lesson is not "add an enum member". It is that the bug was invisible in every
aggregate number I had — gross, incremental, lift, CI all looked completely healthy
— and only became visible when I rendered a single decision as text a human could
read. Aggregates hide state-machine bugs by construction. Two regression tests now
pin it.

### Razorpay integration — webhooks and client

Webhook verification is the trust boundary: everything downstream treats webhook
contents as fact, so getting past it means an attacker can make Recoup believe an
invoice was paid, or fabricate failures to drive messages at arbitrary phone
numbers. Wrote the tests as attacks rather than happy-path coverage.

Four rules, all routinely got wrong. Verify against the **raw body**, never
re-serialised JSON (`json.loads` → `json.dumps` doesn't round-trip byte-for-byte —
the API takes `bytes` not `str` so a caller can't hand over a re-encoded body by
accident, and there's a test pinning that round-tripped bodies *fail* so nobody
"fixes" it by loosening the check). Constant-time compare. Reject stale events —
a valid signature stays valid forever, so without a freshness window a captured
`payment.failed` is a reusable weapon. Reject duplicate event ids, separately,
because Razorpay retries on non-2xx and at-least-once delivery is normal operation
rather than an attack.

Signature is checked before the body is parsed as JSON. Parsing untrusted bytes
before authenticating them hands an attacker your JSON parser.

### The constraint that shaped the client

Went looking for Razorpay's idempotency semantics rather than assuming them, and
the answer changes the design: `X-Payout-Idempotency` covers **only** Create Payout
and the Composite APIs, plus the idempotent Refund and Route variants. Orders,
Payment Links and Subscription charges have **no server-side idempotency at all**.

So a retried POST creates a second order or a second payment link. In a recovery
system that is a double charge against someone who already paid.

That forces the interesting design decision: **a timeout is not a failure, it is an
unknown.** The request may have been processed and the response lost coming back.
Blind-retrying causes duplicate charges; treating it as failure loses money
silently. Both are wrong, so mutating calls that end ambiguously raise
`UncertainOutcome` and the caller *must* reconcile by looking the entity up by a
receipt chosen in advance. That is why `receipt` is a required argument rather than
optional — choosing it after the timeout is too late.

Corollary: 5xx and 429 *are* retried on POST (Razorpay is telling us it never
processed the request, so a retry cannot duplicate), but 408 is not, because a
gateway timeout carries the same ambiguity as a client-side one.

Also: payment links are created with Razorpay's own SMS/email notifications
**disabled**. Letting the payment processor send its own message would route
straight around consent, DND, quiet hours and the contact budget. If a message goes
out it goes through the gated path or it does not go out.

### A one-line bug that would have double-charged customers

The shared-idempotency-store test failed: two clients sharing a store still made
two API calls. Cause:

```python
self.idempotency = idempotency or IdempotencyStore()
```

`IdempotencyStore` defines `__len__`, so an **empty store is falsy** and the
caller's store was silently replaced by a fresh one. The scenario this breaks is
exactly the one it exists for: a workflow resuming on another worker passes in a
store that is empty of everything except the key that must not fire twice.

`X or Y()` is idiomatic enough that I wrote it without thinking. It is only safe
when `X` cannot be falsy, and any object defining `__len__` or `__bool__` can be.
Grepped the rest of the codebase for the same pattern — `ReplayGuard` also defines
`__len__` but is never used that way. Fixed to an explicit `is None` check.

### Wiring the execution path end to end

`engine.py` carries one at-risk rupee through every stage and emits one ledger
record explaining the journey. The structure that mattered was splitting decision
from authorisation:

- `decide()` is pure and authorises **nothing** — diagnosis plus policy, no clock.
- `authorise()` runs all nine gates against the clock **at the moment of action**.
- `execute()` is the only code permitted a side effect, and it never re-decides.

Splitting those two is what makes the multi-day case correct instead of
accidentally correct for same-second execution. A workflow that slept four days may
have been planned inside quiet hours and woken outside them; the customer may have
revoked consent, opened a dispute or promised to pay while it slept. Gate results
computed at plan time and executed later is the standard way a compliant system
emits non-compliant messages.

Two more decisions worth recording:

**The holdout is enforced at the top of the pipeline, not inside policy.** A control
event still runs through diagnosis and policy so the intent is recorded, then is
forced to `NO_ACTION`. The ledger therefore shows what we *would* have done, which
is what makes the counterfactual interpretable rather than a hole in the data. The
demo output shows a holdout event where every gate returned `allow` — the holdout is
what stopped it. "Compliance said no" and "this one is the control" are different
reasons for doing nothing and must not look identical in the record.

**Settlement appends rather than edits.** When a `payment.captured` webhook lands
later, the outcome is a *new* record pointing at the old one's hash. Editing the
original would break the chain — which is exactly the property that makes the chain
worth having.

**A message is never sent if the payment link outcome is unknown.** If the link
POST times out we have nothing to put in the message, and telling somebody to pay
without saying how is worse than silence. `UNRESOLVED`, no contact, human looks.

### The demo

`python -m recoup` runs the whole path with no configuration and no network — a
stateful in-process double stands in for Razorpay, so the reconciliation path is
demonstrable offline. With keys in the environment the same engine drives real test
mode; only the transport changes.

The number I did not expect to care about: **₹10,21,764 of value we chose not to
chase**, blocked by quiet hours (103), consent (29), DND (20) and fatigue (5) on a
300-event run. Compliance is a cost, and printing it is how it stays real. A system
that only reports what it recovered will drift toward messaging more, because the
cost side never appears on the page.

### Tier 2, and getting the safety policy wrong first

Built model-backed diagnosis for codes tier 1 does not have. Wrote what I thought
was the safety policy — one confidence threshold at 0.75, below which nothing
happens — then wired it into the engine, and the integration test failed: an
unmapped code that clearly meant "insufficient funds" produced *no diagnosis at
all*, because the stub's 0.55 confidence sat under the bar.

My first instinct was to lower the threshold. That would have been the wrong fix,
and thinking about why exposed that the design was wrong rather than the number.

A single scalar threshold treats all consequences as equally risky. They are not:

| consequence | customer impact | cost of being wrong |
|---|---|---|
| route to ops | none | an ops ticket nobody needed |
| silent retry | near-zero | one wasted API call |
| contact a customer | high, irreversible | we dun a real person over our own bug |

One threshold got *both* ends wrong simultaneously. It blocked harmless retries on
mediocre-confidence codes, losing recoverable money for no safety benefit — and it
still leaned on confidence for the one decision where confidence is not admissible
evidence, because a model can be confidently wrong and its self-assessment is
exactly the thing you cannot verify at 3am.

The rewritten policy: **tier 2 may route and may retry; it may never authorise
contact, at any confidence including 0.99.** The gate on contact is a human
approving the mined rule, at which point the code becomes tier 1 and contact unlocks
through the ordinary path having been seen by a person. Confidence now only decides
whether a silent retry is worth one API call, and the floor for that is 0.5 — low on
purpose.

There is a parametrised test sweeping the entire confidence range asserting that no
value whatsoever produces a contactable tier-2 diagnosis, so nobody can reintroduce
a threshold there without it failing loudly.

**Every escalation mines a rule.** Each answer is also a candidate row for
`error_taxonomy.tsv`, ranked by how often the code was seen so a reviewer fixes the
most expensive one first. Approving one promotes the code to tier 1 forever: free,
instant, deterministic, auditable. The model's job is to shrink its own job — a
system that calls a model for the same unknown code on the hundred-thousandth
occurrence has not learned anything.

**Caching is a consistency argument before it is a cost argument.** One unknown code
costs one call regardless of how many events carry it. Two identical failures must
never receive different diagnoses because the sampler went a different way. Failures
are cached too, so a model outage gets asked once, not once per event.

Tier 2 is an enhancement, never a dependency: any backend failure degrades to "route
to a human", which is exactly what happens with no tier 2 configured at all.

### Tomorrow

Propensity model and the bandit. Then the ops console (Node is installed now), which
is what makes the approval queue and the review queue actually drainable.

### Open / not yet done

- Tier 2 LLM escalation for unmapped codes is stubbed as `ROUTE_TO_OPS`, not built.
- No bandit yet; `taxonomy_policy` is pure deterministic routing.
- Gate thresholds cite RBI FPC / TRAI / DPDP but need a proper source-by-source review.
- No durable execution yet; the harness runs in-process.
- The approval queue is now visible but nothing drains it — no reviewer UI yet.
- Ledger is JSONL; Postgres later means an append-only table plus these same hashes.
