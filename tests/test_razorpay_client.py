"""Client tests, driven through a mock transport.

The interesting cases are all failure cases: what happens on a timeout mid-POST,
what happens on a 500, and what must NEVER happen (a second charge).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from recoup.razorpay.client import (
    ApiError,
    AuthError,
    IdempotencyStore,
    LiveModeRefused,
    RazorpayClient,
    UncertainOutcome,
    receipt_for,
)

KEY_ID = "rzp_test_1DP5mmOlF5G5ag"
KEY_SECRET = "thisisatestsecret"


class Recorder:
    """Scripted transport. Records every request that actually left the client."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {request.method} {request.url}")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def calls(self) -> int:
        return len(self.requests)


def _json(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status, content=json.dumps(body), headers={"content-type": "application/json"}
    )


def _error(status: int, code: str, description: str) -> httpx.Response:
    return _json(status, {"error": {"code": code, "description": description}})


def _client(
    responses: list[httpx.Response | Exception], **kw: Any
) -> tuple[RazorpayClient, Recorder]:
    rec = Recorder(responses)
    client = RazorpayClient(
        KEY_ID,
        KEY_SECRET,
        transport=httpx.MockTransport(rec.handler),
        max_attempts=kw.pop("max_attempts", 3),
        **kw,
    )
    return client, rec


ORDER_OK = {
    "id": "order_EKwxwAgItmmXdp",
    "entity": "order",
    "amount": 249900,
    "currency": "INR",
    "receipt": "rcp-order-evt_0001-1",
    "status": "created",
}


# --- safety rails -----------------------------------------------------------


def test_live_key_is_refused_by_default() -> None:
    """A buildathon project must not be one env var from moving real money."""
    with pytest.raises(LiveModeRefused, match="refusing live key"):
        RazorpayClient("rzp_live_abc123def456", "secret")


def test_live_key_allowed_with_explicit_optin() -> None:
    client = RazorpayClient("rzp_live_abc123def456", "secret", allow_live=True)
    assert client.is_test_mode is False
    client.close()


def test_missing_credentials_rejected() -> None:
    with pytest.raises(AuthError):
        RazorpayClient("", "")


def test_secret_never_appears_in_repr_or_errors() -> None:
    client, _ = _client([])
    assert KEY_SECRET not in repr(client)
    assert "thisisatest" not in repr(client)
    with pytest.raises(LiveModeRefused) as exc:
        RazorpayClient("rzp_live_supersecretkey", "secret")
    assert "supersecretkey" not in str(exc.value)


# --- the uncertain-outcome contract -----------------------------------------


def test_timeout_on_post_raises_uncertain_not_failure() -> None:
    """A timeout means UNKNOWN. Reporting failure loses money; retrying duplicates
    a charge. Neither is acceptable, so the caller is forced to reconcile."""
    client, rec = _client([httpx.ReadTimeout("timed out")])
    with pytest.raises(UncertainOutcome) as exc:
        client.create_order(249900, receipt="rcp-order-evt_0001-1")
    assert exc.value.receipt == "rcp-order-evt_0001-1"
    assert "Reconcile before retrying" in str(exc.value)
    assert rec.calls == 1, "a timed-out POST must not be retried"


def test_connection_error_on_post_is_also_uncertain() -> None:
    client, rec = _client([httpx.ConnectError("connection reset")])
    with pytest.raises(UncertainOutcome):
        client.create_order(1000, receipt="rcp-order-evt_0002-1")
    assert rec.calls == 1


def test_gateway_408_is_uncertain_not_retried() -> None:
    """408 leaves the same ambiguity as a client-side timeout."""
    client, rec = _client([_error(408, "GATEWAY_ERROR", "request timeout")])
    with pytest.raises(UncertainOutcome):
        client.create_order(1000, receipt="rcp-order-evt_0003-1")
    assert rec.calls == 1


def test_reconcile_finds_an_order_that_did_go_through() -> None:
    """The recovery path after an UncertainOutcome."""
    client, _ = _client([_json(200, {"entity": "collection", "count": 1, "items": [ORDER_OK]})])
    found = client.reconcile_order("rcp-order-evt_0001-1")
    assert found is not None
    assert found["id"] == "order_EKwxwAgItmmXdp"


def test_reconcile_returns_none_when_nothing_was_created() -> None:
    """Only after None is it safe to try again."""
    client, _ = _client([_json(200, {"entity": "collection", "count": 0, "items": []})])
    assert client.reconcile_order("rcp-order-evt_0099-1") is None


# --- retries ----------------------------------------------------------------


def test_post_retries_on_500_because_it_was_not_processed() -> None:
    client, rec = _client([_error(500, "SERVER_ERROR", "internal"), _json(200, ORDER_OK)])
    order = client.create_order(249900, receipt="rcp-order-evt_0001-1")
    assert order["id"] == "order_EKwxwAgItmmXdp"
    assert rec.calls == 2


def test_post_never_retries_on_4xx() -> None:
    """A bad request will be just as bad the second time."""
    client, rec = _client([_error(400, "BAD_REQUEST_ERROR", "amount must be at least 100")])
    with pytest.raises(ApiError) as exc:
        client.create_order(1, receipt="rcp-order-evt_0004-1")
    assert exc.value.status == 400
    assert rec.calls == 1


def test_401_raises_auth_error_and_does_not_retry() -> None:
    client, rec = _client([_error(401, "UNAUTHORIZED", "bad key")])
    with pytest.raises(AuthError):
        client.fetch_payment("pay_x")
    assert rec.calls == 1


def test_get_retries_on_transport_error() -> None:
    """Reads are safe to retry as often as we like."""
    client, rec = _client([httpx.ConnectError("reset"), _json(200, ORDER_OK)])
    assert client.fetch_order("order_x")["id"] == ORDER_OK["id"]
    assert rec.calls == 2


def test_retries_are_bounded() -> None:
    client, rec = _client([_error(503, "SERVER_ERROR", "down")] * 3, max_attempts=3)
    with pytest.raises(ApiError):
        client.fetch_order("order_x")
    assert rec.calls == 3


# --- client-side idempotency ------------------------------------------------


def test_same_receipt_does_not_produce_a_second_order() -> None:
    """Razorpay offers no server-side idempotency here, so this is our job.
    Calling twice must hit the API once."""
    client, rec = _client([_json(200, ORDER_OK)])
    first = client.create_order(249900, receipt="rcp-order-evt_0001-1")
    second = client.create_order(249900, receipt="rcp-order-evt_0001-1")
    assert first == second
    assert rec.calls == 1, "second call must be served from the idempotency store"


def test_different_receipts_do_produce_separate_orders() -> None:
    client, rec = _client([_json(200, ORDER_OK), _json(200, {**ORDER_OK, "id": "order_two"})])
    client.create_order(249900, receipt="rcp-order-evt_0001-1")
    client.create_order(249900, receipt="rcp-order-evt_0001-2")
    assert rec.calls == 2


def test_idempotency_store_can_be_shared_across_clients() -> None:
    """A workflow that restarts on another worker must not re-charge."""
    store = IdempotencyStore()
    c1, r1 = _client([_json(200, ORDER_OK)], idempotency=store)
    c1.create_order(249900, receipt="rcp-order-evt_0001-1")
    c2, r2 = _client([], idempotency=store)
    c2.create_order(249900, receipt="rcp-order-evt_0001-1")
    assert r1.calls == 1
    assert r2.calls == 0


def test_receipt_is_deterministic_and_within_length_limit() -> None:
    a = receipt_for("evt_000123", 2)
    assert a == receipt_for("evt_000123", 2)
    assert a != receipt_for("evt_000123", 3)
    assert len(receipt_for("evt_" + "x" * 100, 1)) <= 40


# --- payment links ----------------------------------------------------------


def test_payment_link_does_not_let_razorpay_notify_the_customer() -> None:
    """Razorpay's own SMS/email would route around consent, DND, quiet hours and
    the contact budget. Contact happens through the gated path or not at all."""
    client, rec = _client([_json(200, {"id": "plink_x", "short_url": "https://rzp.io/i/x"})])
    client.create_payment_link(
        249900,
        reference_id="rcp-link-evt_0001-1",
        description="Renewal",
        customer_contact="+919999999999",
    )
    body = json.loads(rec.requests[0].content)
    assert body["notify"] == {"sms": False, "email": False}
    assert body["reminder_enable"] is False


def test_payment_link_carries_the_reference_for_reconciliation() -> None:
    client, rec = _client([_json(200, {"id": "plink_x"})])
    client.create_payment_link(1000, reference_id="rcp-link-evt_0007-1", description="x")
    assert json.loads(rec.requests[0].content)["reference_id"] == "rcp-link-evt_0007-1"


# --- the seam into diagnosis ------------------------------------------------


def test_polled_payment_failure_matches_the_webhook_shape() -> None:
    """A decision made from a poll must be indistinguishable from one made from a
    webhook, or the two paths drift apart."""
    from recoup.diagnosis.taxonomy import diagnose

    client, _ = _client(
        [
            _json(
                200,
                {
                    "id": "pay_x",
                    "order_id": "order_x",
                    "method": "card",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "card expired",
                    "error_source": "customer",
                    "error_step": "payment_authorization",
                    "error_reason": "card_expired",
                },
            )
        ]
    )
    failure = client.payment_failure("pay_x")
    diagnosis = diagnose(str(failure["error_reason"]))
    assert diagnosis is not None
    assert diagnosis.root_cause == "INSTRUMENT_INVALID"
    assert diagnosis.new_instrument
