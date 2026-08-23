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

### Tomorrow

Razorpay test-mode integration (orders, payment links, subscriptions, webhooks) for
the execution path. The simulator stays for the measurement path — test mode can
produce a failure but it cannot give you the counterfactual, and those are two
different questions.

### Open / not yet done

- Tier 2 LLM escalation for unmapped codes is stubbed as `ROUTE_TO_OPS`, not built.
- No bandit yet; `taxonomy_policy` is pure deterministic routing.
- Gate thresholds cite RBI FPC / TRAI / DPDP but need a proper source-by-source review.
- No durable execution yet; the harness runs in-process.
