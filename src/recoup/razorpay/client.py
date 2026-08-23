"""Razorpay test-mode API client for the execution path.

Scope is deliberately small: the handful of endpoints a recovery workflow needs to
actually move money -- create an order, raise a payment link, re-charge a
subscription, and read back what happened.

Three things drive the design, and the first is a constraint Razorpay imposes:

**There is no server-side idempotency on the endpoints we use.**
`X-Payout-Idempotency` exists, but per the docs it covers only Create Payout and the
Composite APIs (plus the idempotent Refund and Route Direct Transfer variants).
Orders, Payment Links and Subscription charges have none. So a retried POST creates
a SECOND order or a SECOND payment link, and in a recovery system that means
double-charging a customer who already paid.

Recoup therefore provides idempotency itself, and the important half is not the
local cache -- it is what happens on a timeout.

**A timeout is not a failure. It is an unknown.**
The request may have been fully processed and the response lost on the way back.
Blind-retrying is how duplicate charges happen; treating it as a failure is how
money silently goes missing. Both are wrong. Mutating calls that end in an unknown
state raise `UncertainOutcome`, which the caller MUST resolve by reconciling
against the API -- looking the entity up by the receipt we chose in advance --
rather than by guessing. `reconcile_order` does exactly that.

Read-only calls retry freely. Mutating calls never retry blindly.

**Test mode is enforced, not assumed.** The client refuses `rzp_live_` keys unless
`allow_live=True` is passed explicitly. A buildathon project has no business being
one environment variable away from moving real money.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

BASE_URL = "https://api.razorpay.com/v1"
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4


class RazorpayError(Exception):
    """Base class for everything this client raises."""


class AuthError(RazorpayError):
    """Bad credentials. Never retried -- retrying will not fix a wrong key."""


class LiveModeRefused(RazorpayError):
    """A live key was supplied without explicit opt-in."""


class ApiError(RazorpayError):
    """Razorpay returned a 4xx we should not retry."""

    def __init__(self, status: int, code: str, description: str, reason: str | None = None):
        self.status = status
        self.code = code
        self.description = description
        self.reason = reason
        super().__init__(f"{status} {code}: {description}" + (f" ({reason})" if reason else ""))


class UncertainOutcome(RazorpayError):
    """A mutating request may or may not have taken effect.

    Raised on timeout or connection loss during a POST. The caller must reconcile
    before doing anything else -- never retry, and never assume failure. Carries the
    receipt so reconciliation has something to look the entity up by.
    """

    def __init__(self, operation: str, receipt: str | None, cause: Exception):
        self.operation = operation
        self.receipt = receipt
        self.cause = cause
        super().__init__(
            f"{operation} outcome unknown (receipt={receipt!r}): {type(cause).__name__}. "
            "Reconcile before retrying -- the request may have succeeded."
        )


def _redact(value: str) -> str:
    """Secrets must never reach a log line, an exception message or the ledger."""
    if len(value) <= 8:
        return "***"
    return f"{value[:8]}...{'*' * 6}"


@dataclass(slots=True)
class IdempotencyStore:
    """Client-side request dedupe, standing in for the server-side idempotency
    Razorpay does not offer on these endpoints.

    In production this is Redis or a unique index. The contract is what matters:
    the same key must never produce a second live call.
    """

    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, response: dict[str, Any]) -> None:
        self._entries[key] = response

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class RazorpayClient:
    """Thin, typed, and paranoid about the mutating calls."""

    def __init__(
        self,
        key_id: str,
        key_secret: str,
        *,
        base_url: str = BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        allow_live: bool = False,
        transport: httpx.BaseTransport | None = None,
        idempotency: IdempotencyStore | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not key_id or not key_secret:
            raise AuthError("key_id and key_secret are both required")
        if key_id.startswith("rzp_live_") and not allow_live:
            raise LiveModeRefused(
                f"refusing live key {_redact(key_id)} -- pass allow_live=True to override. "
                "Recoup is built and evaluated against test mode."
            )

        self.key_id = key_id
        self._secret = key_secret
        self.is_test_mode = key_id.startswith("rzp_test_")
        # `idempotency or IdempotencyStore()` looks equivalent and is not: this class
        # defines __len__, so an EMPTY store is falsy and the caller's store would be
        # silently replaced by a fresh one. That is a double-charge bug -- a workflow
        # resuming on another worker passes in a store precisely because it is empty
        # of everything except the key that must not fire twice.
        self.idempotency = IdempotencyStore() if idempotency is None else idempotency
        self._max_attempts = max_attempts
        self._client = httpx.Client(
            base_url=base_url,
            auth=(key_id, key_secret),
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "recoup/0.1 (razorpay-buildathon-2026)"},
        )

    def __enter__(self) -> RazorpayClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def __repr__(self) -> str:
        mode = "test" if self.is_test_mode else "LIVE"
        return f"<RazorpayClient {mode} key={_redact(self.key_id)}>"

    # --- transport ----------------------------------------------------------

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        # Exponential with full jitter. Without jitter, a fleet retrying a shared
        # outage re-synchronises and hammers the API in lockstep.
        return random.uniform(0, min(2**attempt * 0.5, 8.0))

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise AuthError(f"authentication failed for key {_redact(self.key_id)}")
        try:
            body = response.json().get("error", {})
        except ValueError:
            body = {}
        raise ApiError(
            status=response.status_code,
            code=str(body.get("code", "UNKNOWN")),
            description=str(body.get("description", response.text[:200])),
            reason=body.get("reason"),
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read-only. Safe to retry as often as we like."""
        last: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                last = exc
                if attempt + 1 >= self._max_attempts:
                    break
                time.sleep(self._backoff(attempt, None))
                continue

            if response.status_code < 300:
                return dict(response.json())
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < self._max_attempts:
                time.sleep(self._backoff(attempt, response.headers.get("Retry-After")))
                continue
            self._raise_for_status(response)
        raise RazorpayError(f"GET {path} failed after {self._max_attempts} attempts: {last}")

    def _post(
        self,
        path: str,
        json_body: dict[str, Any],
        *,
        operation: str,
        receipt: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Mutating. Never retried blindly.

        A 5xx is retryable only because Razorpay has told us it did not process the
        request. A timeout tells us nothing, so it becomes UncertainOutcome and the
        caller reconciles.
        """
        if idempotency_key is not None:
            cached = self.idempotency.get(idempotency_key)
            if cached is not None:
                return cached

        last_status: int | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.post(path, json=json_body)
            except httpx.TransportError as exc:
                # Includes ReadTimeout: the server may well have processed this.
                raise UncertainOutcome(operation, receipt, exc) from exc

            if response.status_code < 300:
                data = dict(response.json())
                if idempotency_key is not None:
                    self.idempotency.put(idempotency_key, data)
                return data

            last_status = response.status_code
            # 5xx and 429 mean it was rejected before processing, so a retry cannot
            # duplicate. 408 is excluded here on purpose -- a request timeout leaves
            # the same ambiguity as a client-side one.
            if (
                response.status_code in (429, 500, 502, 503, 504)
                and attempt + 1 < self._max_attempts
            ):
                time.sleep(self._backoff(attempt, response.headers.get("Retry-After")))
                continue
            if response.status_code == 408:
                raise UncertainOutcome(
                    operation, receipt, RazorpayError("408 request timeout at gateway")
                )
            self._raise_for_status(response)

        raise RazorpayError(
            f"{operation} failed after {self._max_attempts} attempts (last status {last_status})"
        )

    # --- orders -------------------------------------------------------------

    def create_order(
        self,
        amount_paise: int,
        *,
        receipt: str,
        currency: str = "INR",
        notes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create an order.

        `receipt` is required rather than optional: it is the only handle we get for
        reconciling an uncertain outcome, and choosing it AFTER a timeout is too
        late. Callers should derive it deterministically from the recovery attempt
        (e.g. `recoup:{event_id}:{attempt}`) so the same attempt always produces the
        same receipt.
        """
        return self._post(
            "/orders",
            {
                "amount": amount_paise,
                "currency": currency,
                "receipt": receipt,
                "notes": notes or {},
            },
            operation="create_order",
            receipt=receipt,
            idempotency_key=f"order:{receipt}",
        )

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        return self._get(f"/orders/{order_id}")

    def find_order_by_receipt(self, receipt: str, *, count: int = 100) -> dict[str, Any] | None:
        """Look an order up by the receipt we chose. The basis of reconciliation."""
        page = self._get("/orders", {"count": count})
        for item in page.get("items", []):
            if item.get("receipt") == receipt:
                return dict(item)
        return None

    def reconcile_order(self, receipt: str) -> dict[str, Any] | None:
        """Resolve an UncertainOutcome from `create_order`.

        Returns the order if it was in fact created, None if it genuinely was not.
        Only after a None is it safe to try again. This is the whole reason
        `receipt` is mandatory.
        """
        return self.find_order_by_receipt(receipt)

    # --- payment links ------------------------------------------------------

    def create_payment_link(
        self,
        amount_paise: int,
        *,
        reference_id: str,
        description: str,
        customer_name: str | None = None,
        customer_email: str | None = None,
        customer_contact: str | None = None,
        expire_by: int | None = None,
        notify_sms: bool = False,
        notify_email: bool = False,
        notes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Raise a payment link.

        Razorpay can notify the customer itself, but both notify flags default to
        FALSE here. Recoup's compliance gates decide whether a customer may be
        contacted, and letting the payment processor send its own SMS would route
        around consent, DND, quiet hours and the contact budget entirely. If a
        message goes out, it goes out through the gated path or not at all.
        """
        customer: dict[str, str] = {}
        if customer_name:
            customer["name"] = customer_name
        if customer_email:
            customer["email"] = customer_email
        if customer_contact:
            customer["contact"] = customer_contact

        body: dict[str, Any] = {
            "amount": amount_paise,
            "currency": "INR",
            "accept_partial": False,
            "reference_id": reference_id,
            "description": description,
            "notify": {"sms": notify_sms, "email": notify_email},
            "reminder_enable": False,
            "notes": notes or {},
        }
        if customer:
            body["customer"] = customer
        if expire_by is not None:
            body["expire_by"] = expire_by

        return self._post(
            "/payment_links",
            body,
            operation="create_payment_link",
            receipt=reference_id,
            idempotency_key=f"plink:{reference_id}",
        )

    def fetch_payment_link(self, link_id: str) -> dict[str, Any]:
        return self._get(f"/payment_links/{link_id}")

    # --- payments and subscriptions ----------------------------------------

    def fetch_payment(self, payment_id: str) -> dict[str, Any]:
        return self._get(f"/payments/{payment_id}")

    def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
        return self._get(f"/subscriptions/{subscription_id}")

    def fetch_invoice(self, invoice_id: str) -> dict[str, Any]:
        return self._get(f"/invoices/{invoice_id}")

    def payment_failure(self, payment_id: str) -> dict[str, str | None]:
        """The fields diagnosis needs, pulled from a fetched payment.

        Same shape as `webhooks.extract_failure`, so a recovery decision made from a
        webhook and one made from a poll are indistinguishable downstream.
        """
        payment = self.fetch_payment(payment_id)
        return {
            "payment_id": payment.get("id"),
            "order_id": payment.get("order_id"),
            "method": payment.get("method"),
            "error_code": payment.get("error_code"),
            "error_description": payment.get("error_description"),
            "error_source": payment.get("error_source"),
            "error_step": payment.get("error_step"),
            "error_reason": payment.get("error_reason"),
        }


def receipt_for(event_id: str, attempt: int, kind: Literal["order", "link"] = "order") -> str:
    """Deterministic receipt for a recovery attempt.

    Same event and attempt always yields the same receipt, which is what makes
    reconciliation possible and what stops a retried workflow raising a second
    charge. Capped at Razorpay's 40-character limit for receipts.
    """
    return f"rcp-{kind}-{event_id}-{attempt}"[:40]
