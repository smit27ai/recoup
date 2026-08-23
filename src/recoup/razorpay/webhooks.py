"""Razorpay webhook verification and event parsing.

This is the trust boundary. Everything downstream -- diagnosis, policy, execution,
the ledger -- treats webhook contents as fact, so if this module is wrong then an
attacker who can reach the endpoint can make Recoup believe an invoice was paid, or
fabricate failures to trigger recovery messages to arbitrary phone numbers.

Four rules it enforces, all of which are routinely got wrong:

1. **Verify against the RAW BODY, never re-serialised JSON.** `json.loads` followed
   by `json.dumps` does not round-trip byte-for-byte: key order, unicode escaping
   and float formatting can all shift, and the HMAC then fails or, far worse, an
   implementation "fixes" it by loosening the check. The raw bytes as received are
   the only thing the signature covers.

2. **Constant-time comparison.** `==` on the digest leaks timing information that
   can be used to forge a signature byte by byte. `hmac.compare_digest` exists
   precisely for this.

3. **Reject stale events.** A valid signature stays valid forever. Without a
   freshness window, an attacker who captures one legitimate `payment.failed`
   webhook can replay it indefinitely and drive repeated recovery attempts against
   a real customer.

4. **Reject duplicate event ids.** Razorpay retries webhooks on non-2xx, so
   at-least-once delivery is normal operation, not an attack. Processing the same
   event twice means charging or messaging twice.

Rules 3 and 4 are separate on purpose: 3 is about an adversary, 4 is about ordinary
delivery semantics, and a system needs both.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

MAX_EVENT_AGE = timedelta(minutes=5)
"""How old a webhook may be before we refuse it. Generous enough to survive a slow
retry, short enough that a captured payload is not a reusable weapon."""


class WebhookError(Exception):
    """Base class. Every subclass means: do not act on this payload."""


class SignatureMismatch(WebhookError):
    """The signature does not verify. Treat the payload as hostile, not as corrupt."""


class StaleEvent(WebhookError):
    """Correctly signed but too old -- a replay of a once-legitimate event."""


class DuplicateEvent(WebhookError):
    """Already processed. Normal under at-least-once delivery; must be a no-op."""


class MalformedEvent(WebhookError):
    """Signature verified but the body is not a webhook we understand."""


def compute_signature(body: bytes, secret: str) -> str:
    """HMAC-SHA256 of the raw body, hex encoded -- what Razorpay sends in
    `X-Razorpay-Signature`."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_signature(body: bytes, signature: str, secret: str) -> None:
    """Raise SignatureMismatch unless `signature` covers exactly these bytes.

    Takes `bytes`, not `str`, so a caller cannot accidentally hand us a decoded and
    re-encoded body. The type is the guard rail.
    """
    if not signature:
        raise SignatureMismatch("no signature header present")
    expected = compute_signature(body, secret)
    if not hmac.compare_digest(expected, signature):
        raise SignatureMismatch("signature does not match request body")


def verify_payment_signature(
    order_id: str, payment_id: str, signature: str, key_secret: str
) -> None:
    """Verify a Checkout handler callback.

    Different construction from webhooks: the signed message is
    `{order_id}|{payment_id}` and the key is the API secret, not the webhook secret.
    Mixing the two up is a common integration bug and fails closed here.
    """
    message = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(key_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise SignatureMismatch("payment signature does not match order/payment pair")


@dataclass(frozen=True, slots=True)
class WebhookEvent:
    """A verified Razorpay webhook."""

    event_id: str
    event: str
    """e.g. payment.failed, payment.captured, subscription.halted, invoice.paid."""
    created_at: datetime
    payload: dict[str, Any]
    raw: bytes = field(repr=False, default=b"")

    def entity(self, name: str) -> dict[str, Any]:
        """Pull one entity out of the payload envelope.

        Razorpay nests as payload.<entity>.entity, which is easy to fumble and
        produces confusing failures three layers away from the mistake.
        """
        try:
            return dict(self.payload[name]["entity"])
        except (KeyError, TypeError) as exc:
            raise MalformedEvent(f"no {name!r} entity in {self.event} payload") from exc

    @property
    def is_failure(self) -> bool:
        return self.event in {
            "payment.failed",
            "subscription.halted",
            "subscription.pending",
            "invoice.expired",
            "order.paid.failed",
        }

    @property
    def is_recovery(self) -> bool:
        """The money arrived. Recovery workflows for this event must stop now."""
        return self.event in {
            "payment.captured",
            "payment.authorized",
            "invoice.paid",
            "subscription.charged",
            "order.paid",
        }


class ReplayGuard:
    """Rejects events already seen. In-memory; Redis or a unique index in production.

    Bounded on purpose -- an unbounded set of every event id ever seen is a memory
    leak that only shows up in production, at volume, weeks in.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._seen: dict[str, datetime] = {}
        self._capacity = capacity

    def check_and_record(self, event_id: str, now: datetime | None = None) -> None:
        moment = now or datetime.now(UTC)
        if event_id in self._seen:
            raise DuplicateEvent(f"event {event_id} already processed")
        if len(self._seen) >= self._capacity:
            # Evict oldest. Cheap because this runs once per capacity-worth of events.
            for stale, _ in sorted(self._seen.items(), key=lambda kv: kv[1])[
                : self._capacity // 10
            ]:
                del self._seen[stale]
        self._seen[event_id] = moment

    def __contains__(self, event_id: object) -> bool:
        return event_id in self._seen

    def __len__(self) -> int:
        return len(self._seen)


def parse(
    body: bytes,
    signature: str,
    secret: str,
    *,
    now: datetime | None = None,
    max_age: timedelta | None = MAX_EVENT_AGE,
    replay_guard: ReplayGuard | None = None,
) -> WebhookEvent:
    """Verify and parse a webhook, in the only order that is safe.

    Signature FIRST, before the body is even parsed as JSON. Parsing untrusted bytes
    before authenticating them hands an attacker your JSON parser.
    """
    verify_signature(body, signature, secret)

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedEvent("signed body is not valid JSON") from exc
    if not isinstance(data, dict):
        raise MalformedEvent("webhook body is not a JSON object")

    event_name = data.get("event")
    if not isinstance(event_name, str) or not event_name:
        raise MalformedEvent("webhook has no event name")

    created = data.get("created_at")
    if not isinstance(created, int):
        raise MalformedEvent("webhook has no integer created_at")
    created_at = datetime.fromtimestamp(created, tz=UTC)

    moment = now or datetime.now(UTC)
    if max_age is not None:
        age = moment - created_at
        if age > max_age:
            raise StaleEvent(
                f"event is {age.total_seconds():.0f}s old, limit is "
                f"{max_age.total_seconds():.0f}s -- possible replay"
            )
        # Clocks drift, but a webhook from the future by more than a small margin
        # means either a badly wrong clock or a forged timestamp. Neither is safe.
        if created_at - moment > timedelta(minutes=1):
            raise StaleEvent("event timestamp is in the future")

    event_id = data.get("id") or _derive_event_id(body)
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise MalformedEvent("webhook has no payload object")

    if replay_guard is not None:
        replay_guard.check_and_record(event_id, moment)

    return WebhookEvent(
        event_id=event_id,
        event=event_name,
        created_at=created_at,
        payload=payload,
        raw=body,
    )


def _derive_event_id(body: bytes) -> str:
    """Fallback id for payloads that carry none.

    Content-addressed, so two byte-identical deliveries collapse to one id and the
    replay guard still works. Deliberately not random: a random id would make every
    retry look like a new event, which is the exact failure the guard prevents.
    """
    return "sha256:" + hashlib.sha256(body).hexdigest()[:32]


FAILURE_FIELDS = ("error_code", "error_description", "error_source", "error_step", "error_reason")


def extract_failure(event: WebhookEvent) -> dict[str, str | int | None]:
    """Pull the fields Recoup diagnosis needs out of a payment.failed webhook.

    `error_reason` is the one that matters -- it is the key into the taxonomy. The
    rest are kept for the ledger so a reviewer can see the whole gateway response
    rather than our interpretation of it.
    """
    entity = event.entity("payment")
    out: dict[str, str | int | None] = {
        "payment_id": entity.get("id"),
        "order_id": entity.get("order_id"),
        "amount_paise": entity.get("amount"),
        "currency": entity.get("currency"),
        "method": entity.get("method"),
    }
    for key in FAILURE_FIELDS:
        value = entity.get(key)
        out[key] = value if isinstance(value, str | int) or value is None else str(value)
    return out


def dedupe(events: Iterable[WebhookEvent]) -> list[WebhookEvent]:
    """Collapse a batch to first-seen-wins, preserving order."""
    seen: set[str] = set()
    out: list[WebhookEvent] = []
    for ev in events:
        if ev.event_id not in seen:
            seen.add(ev.event_id)
            out.append(ev)
    return out
