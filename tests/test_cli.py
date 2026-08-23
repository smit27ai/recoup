"""Smoke tests for the demo entrypoint.

A reviewer clones the repo and runs one command. If that command is broken, none
of the rest of the work is visible to them.
"""

from __future__ import annotations

import json

import httpx

from recoup.cli import LocalRazorpay, build_client, demo


def test_demo_runs_with_no_configuration(capsys, monkeypatch) -> None:
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    assert demo(n_events=60) == 0
    out = capsys.readouterr().out
    assert "hash chain verified" in out
    assert "holdout contacted   0" in out


def test_falls_back_to_local_double_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    _, mode = build_client()
    assert "local in-process double" in mode


def test_uses_real_client_when_keys_are_present(monkeypatch) -> None:
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_abcdef123456")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    client, mode = build_client()
    assert "Razorpay test mode" in mode
    assert client.is_test_mode
    client.close()


def test_live_key_in_env_is_still_refused(monkeypatch) -> None:
    """The env var must not be a way around the live-mode guard."""
    import pytest

    from recoup.razorpay.client import LiveModeRefused

    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abcdef123456")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    with pytest.raises(LiveModeRefused):
        build_client()


def test_local_double_is_idempotent_on_receipt() -> None:
    """Statefulness is what makes the reconciliation path demonstrable offline."""
    fake = LocalRazorpay()
    req = httpx.Request(
        "POST",
        "https://api.razorpay.com/v1/orders",
        content=json.dumps({"amount": 1000, "receipt": "rcp-1"}),
    )
    first = json.loads(fake.handler(req).content)
    second = json.loads(fake.handler(req).content)
    assert first["id"] == second["id"]
    assert len(fake.orders) == 1
