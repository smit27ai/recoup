"""Webhook verification tests.

This is the trust boundary, so the tests are written as attacks rather than as
happy-path coverage. Anything that gets past these can make Recoup act on a lie.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from recoup.diagnosis.taxonomy import diagnose
from recoup.razorpay.webhooks import (
    DuplicateEvent,
    MalformedEvent,
    ReplayGuard,
    SignatureMismatch,
    StaleEvent,
    compute_signature,
    dedupe,
    extract_failure,
    parse,
    verify_payment_signature,
    verify_signature,
)

SECRET = "whsec_test_5f2c9a1b7e"
NOW = datetime(2026, 9, 1, 11, 0, tzinfo=UTC)


def _body(
    event: str = "payment.failed",
    *,
    created_at: datetime = NOW,
    event_id: str = "evt_QpX1mN2oP3qR4s",
    error_reason: str = "insufficient_funds",
    amount: int = 249900,
) -> bytes:
    """A payload shaped like a real Razorpay webhook."""
    payload = {
        "entity": "event",
        "id": event_id,
        "account_id": "acc_test_Jk9LmN0pQr",
        "event": event,
        "contains": ["payment"],
        "created_at": int(created_at.timestamp()),
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_QpX1mN2oP3qR4s",
                    "entity": "payment",
                    "amount": amount,
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_QpX1mN2oP3qR4s",
                    "method": "card",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Your account does not have enough balance.",
                    "error_source": "customer",
                    "error_step": "payment_authorization",
                    "error_reason": error_reason,
                }
            }
        },
    }
    return json.dumps(payload).encode("utf-8")


def _sign(body: bytes) -> str:
    return compute_signature(body, SECRET)


# --- signature verification -------------------------------------------------


def test_valid_signature_passes() -> None:
    body = _body()
    verify_signature(body, _sign(body), SECRET)


def test_tampered_body_fails() -> None:
    """Change one rupee after signing and it must not verify."""
    body = _body(amount=249900)
    sig = _sign(body)
    tampered = body.replace(b'"amount": 249900', b'"amount": 100')
    assert tampered != body
    with pytest.raises(SignatureMismatch):
        verify_signature(tampered, sig, SECRET)


def test_tampered_error_reason_fails() -> None:
    """The field that drives the whole policy must be signature-protected."""
    body = _body(error_reason="insufficient_funds")
    sig = _sign(body)
    tampered = body.replace(b"insufficient_funds", b"gateway_technical_err")
    with pytest.raises(SignatureMismatch):
        verify_signature(tampered, sig, SECRET)


def test_wrong_secret_fails() -> None:
    body = _body()
    with pytest.raises(SignatureMismatch):
        verify_signature(body, compute_signature(body, "whsec_wrong"), SECRET)


def test_empty_signature_fails() -> None:
    with pytest.raises(SignatureMismatch, match="no signature"):
        verify_signature(_body(), "", SECRET)


def test_signature_from_a_different_event_fails() -> None:
    """Lifting a valid signature onto another payload must not work."""
    other = _body(event="payment.captured", event_id="evt_other")
    with pytest.raises(SignatureMismatch):
        verify_signature(_body(), _sign(other), SECRET)


def test_reserialised_body_would_break_verification() -> None:
    """Documents WHY the API takes raw bytes.

    A caller who parses then re-dumps produces different bytes, and the signature
    correctly stops matching. This test exists so nobody 'fixes' that by relaxing
    the check.
    """
    body = _body()
    sig = _sign(body)
    round_tripped = json.dumps(json.loads(body), separators=(",", ":")).encode()
    assert round_tripped != body
    with pytest.raises(SignatureMismatch):
        verify_signature(round_tripped, sig, SECRET)


# --- checkout payment signature ---------------------------------------------


def test_payment_signature_valid() -> None:
    import hashlib
    import hmac

    key_secret = "rzp_test_secret_abc123"
    order_id, payment_id = "order_ABC123", "pay_XYZ789"
    sig = hmac.new(
        key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256
    ).hexdigest()
    verify_payment_signature(order_id, payment_id, sig, key_secret)


def test_payment_signature_rejects_swapped_ids() -> None:
    import hashlib
    import hmac

    key_secret = "rzp_test_secret_abc123"
    sig = hmac.new(key_secret.encode(), b"order_ABC123|pay_XYZ789", hashlib.sha256).hexdigest()
    with pytest.raises(SignatureMismatch):
        verify_payment_signature("pay_XYZ789", "order_ABC123", sig, key_secret)


def test_webhook_secret_does_not_validate_payment_signature() -> None:
    """Mixing up the two secrets is a classic integration bug. It must fail closed."""
    body = _body()
    with pytest.raises(SignatureMismatch):
        verify_payment_signature("order_A", "pay_B", _sign(body), SECRET)


# --- replay and staleness ---------------------------------------------------


def test_stale_event_rejected() -> None:
    """A captured webhook must not stay usable forever."""
    old = NOW - timedelta(hours=2)
    body = _body(created_at=old)
    with pytest.raises(StaleEvent, match="possible replay"):
        parse(body, _sign(body), SECRET, now=NOW)


def test_event_within_window_accepted() -> None:
    body = _body(created_at=NOW - timedelta(minutes=2))
    assert parse(body, _sign(body), SECRET, now=NOW).event == "payment.failed"


def test_future_timestamp_rejected() -> None:
    body = _body(created_at=NOW + timedelta(hours=1))
    with pytest.raises(StaleEvent, match="future"):
        parse(body, _sign(body), SECRET, now=NOW)


def test_duplicate_event_id_rejected() -> None:
    """At-least-once delivery is normal; acting twice is not."""
    guard = ReplayGuard()
    body = _body()
    sig = _sign(body)
    parse(body, sig, SECRET, now=NOW, replay_guard=guard)
    with pytest.raises(DuplicateEvent):
        parse(body, sig, SECRET, now=NOW, replay_guard=guard)


def test_distinct_events_both_accepted() -> None:
    guard = ReplayGuard()
    for eid in ("evt_one", "evt_two", "evt_three"):
        body = _body(event_id=eid)
        parse(body, _sign(body), SECRET, now=NOW, replay_guard=guard)
    assert len(guard) == 3


def test_replay_guard_is_bounded() -> None:
    """An unbounded seen-set is a production-only memory leak."""
    guard = ReplayGuard(capacity=100)
    for i in range(500):
        guard.check_and_record(f"evt_{i}", NOW + timedelta(seconds=i))
    assert len(guard) <= 100


def test_identical_unsigned_bodies_get_the_same_derived_id() -> None:
    """Payloads without an id must still dedupe, or every retry looks new."""
    raw = json.dumps(
        {"event": "payment.failed", "created_at": int(NOW.timestamp()), "payload": {}}
    ).encode()
    guard = ReplayGuard()
    parse(raw, _sign(raw), SECRET, now=NOW, replay_guard=guard)
    with pytest.raises(DuplicateEvent):
        parse(raw, _sign(raw), SECRET, now=NOW, replay_guard=guard)


# --- parsing ----------------------------------------------------------------


def test_signature_is_checked_before_json_parsing() -> None:
    """Unsigned garbage must never reach the JSON parser."""
    garbage = b"{{{ not json at all"
    with pytest.raises(SignatureMismatch):
        parse(garbage, "deadbeef", SECRET, now=NOW)


def test_signed_but_invalid_json_rejected() -> None:
    body = b"not json"
    with pytest.raises(MalformedEvent, match="not valid JSON"):
        parse(body, _sign(body), SECRET, now=NOW)


def test_signed_json_array_rejected() -> None:
    body = b"[1, 2, 3]"
    with pytest.raises(MalformedEvent, match="not a JSON object"):
        parse(body, _sign(body), SECRET, now=NOW)


def test_missing_event_name_rejected() -> None:
    body = json.dumps({"created_at": int(NOW.timestamp()), "payload": {}}).encode()
    with pytest.raises(MalformedEvent, match="no event name"):
        parse(body, _sign(body), SECRET, now=NOW)


def test_missing_created_at_rejected() -> None:
    body = json.dumps({"event": "payment.failed", "payload": {}}).encode()
    with pytest.raises(MalformedEvent, match="created_at"):
        parse(body, _sign(body), SECRET, now=NOW)


def test_missing_entity_raises_a_useful_error() -> None:
    body = json.dumps(
        {
            "event": "payment.failed",
            "id": "evt_x",
            "created_at": int(NOW.timestamp()),
            "payload": {},
        }
    ).encode()
    event = parse(body, _sign(body), SECRET, now=NOW)
    with pytest.raises(MalformedEvent, match="no 'payment' entity"):
        event.entity("payment")


def test_failure_and_recovery_classification() -> None:
    for name, failure, recovery in [
        ("payment.failed", True, False),
        ("subscription.halted", True, False),
        ("payment.captured", False, True),
        ("invoice.paid", False, True),
    ]:
        body = _body(event=name)
        ev = parse(body, _sign(body), SECRET, now=NOW)
        assert ev.is_failure is failure
        assert ev.is_recovery is recovery


# --- the seam into diagnosis ------------------------------------------------


def test_extracted_failure_feeds_the_taxonomy() -> None:
    """End to end: a signed webhook must produce a usable diagnosis."""
    body = _body(error_reason="card_expired")
    event = parse(body, _sign(body), SECRET, now=NOW)
    failure = extract_failure(event)

    assert failure["amount_paise"] == 249900
    assert failure["error_source"] == "customer"

    diagnosis = diagnose(str(failure["error_reason"]))
    assert diagnosis is not None
    assert diagnosis.root_cause == "INSTRUMENT_INVALID"
    assert diagnosis.new_instrument, "retrying an expired card is futile"


def test_unmapped_reason_from_a_real_webhook_escalates() -> None:
    """A code Razorpay adds after we shipped must escalate, not be guessed at."""
    body = _body(error_reason="some_new_2027_reason")
    event = parse(body, _sign(body), SECRET, now=NOW)
    assert diagnose(str(extract_failure(event)["error_reason"])) is None


def test_dedupe_preserves_order() -> None:
    events = []
    for eid in ("a", "b", "a", "c", "b"):
        body = _body(event_id=f"evt_{eid}")
        events.append(parse(body, _sign(body), SECRET, now=NOW))
    assert [e.event_id for e in dedupe(events)] == ["evt_a", "evt_b", "evt_c"]
