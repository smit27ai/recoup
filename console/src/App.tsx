import { useCallback, useEffect, useState } from "react";
import {
  api,
  rupees,
  type Decision,
  type Integrity,
  type Metrics,
  type PendingRule,
  type QueueItem,
} from "./api";

type Tab = "overview" | "decisions" | "approvals" | "rules";

const REVIEWER_KEY = "recoup.reviewer";

function dispositionTag(d: Decision) {
  if (d.arm === "holdout") return <span className="tag holdout">holdout</span>;
  if (d.denied_by.length) return <span className="tag deny">denied</span>;
  if (d.disposition === "needs_approval") return <span className="tag hold">awaiting human</span>;
  return <span className="tag allow">allowed</span>;
}

function Bars({ data, deny }: { data: Record<string, number>; deny?: boolean }) {
  const entries = Object.entries(data);
  if (!entries.length) return <div className="empty">nothing recorded</div>;
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <div className="bars">
      {entries.map(([k, v]) => (
        <div className="bar" key={k}>
          <span className="k">{k}</span>
          <span className="track">
            <span
              className={deny ? "fill deny" : "fill"}
              style={{ width: `${(v / max) * 100}%` }}
            />
          </span>
          <span className="n">{v.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

function Overview({ m }: { m: Metrics }) {
  return (
    <>
      <div className="cards">
        <div className="card">
          <div className="label">decisions</div>
          <div className="value">{m.decisions.toLocaleString()}</div>
          <div className="foot">{m.holdout.toLocaleString()} held out as control</div>
        </div>
        <div className="card">
          <div className="label">at risk</div>
          <div className="value">{rupees(m.at_risk_paise)}</div>
        </div>
        <div className="card warn">
          <div className="label">not chased</div>
          <div className="value">{rupees(m.not_chased_paise)}</div>
          <div className="foot">blocked by compliance gates</div>
        </div>
        <div className="card warn">
          <div className="label">awaiting human</div>
          <div className="value">{rupees(m.approval_queue_paise)}</div>
          <div className="foot">{m.approval_queue} actions parked</div>
        </div>
      </div>

      <div className="panel">
        <h2>What we did</h2>
        <Bars data={m.by_action} />
      </div>

      <div className="panel">
        <h2>Why we did not act</h2>
        <Bars data={m.denials_by_gate} deny />
        <p className="note">
          Compliance is a cost, measured here in rupees. A console that only showed money
          recovered would quietly push its operators toward messaging more.
        </p>
      </div>

      <div className="panel">
        <h2>Diagnosis</h2>
        <div className="cards">
          <div className="card">
            <div className="label">tier-2 calls</div>
            <div className="value">{m.escalation_calls}</div>
            <div className="foot">one per unknown code, not per event</div>
          </div>
          <div className="card">
            <div className="label">rules awaiting review</div>
            <div className="value">{m.pending_rules}</div>
            <div className="foot">approving one makes it tier 1 forever</div>
          </div>
          <div className="card">
            <div className="label">ops queue</div>
            <div className="value">{rupees(m.ops_queue_paise)}</div>
            <div className="foot">{m.ops_queue} not customer-actionable</div>
          </div>
        </div>
      </div>
    </>
  );
}

function Decisions() {
  const [rows, setRows] = useState<Decision[]>([]);
  const [blockedOnly, setBlockedOnly] = useState(false);
  const [tier2Only, setTier2Only] = useState(false);
  const [open, setOpen] = useState<Decision | null>(null);

  useEffect(() => {
    api
      .decisions({ blocked_only: blockedOnly, tier: tier2Only ? 2 : undefined, limit: 200 })
      .then((r) => setRows(r.items))
      .catch(() => setRows([]));
  }, [blockedOnly, tier2Only]);

  return (
    <>
      <div className="controls">
        <label className="check">
          <input
            type="checkbox"
            checked={blockedOnly}
            onChange={(e) => setBlockedOnly(e.target.checked)}
          />
          blocked only
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={tier2Only}
            onChange={(e) => setTier2Only(e.target.checked)}
          />
          tier-2 diagnoses only
        </label>
        <span className="mono" style={{ color: "var(--muted)", fontSize: 12 }}>
          {rows.length} shown
        </span>
      </div>

      <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
        <table>
          <thead>
            <tr>
              <th>event</th>
              <th>saw</th>
              <th>root cause</th>
              <th>wanted</th>
              <th>did</th>
              <th style={{ textAlign: "right" }}>amount</th>
              <th>status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => (
              <tr key={d.record_hash} className="clickable" onClick={() => setOpen(d)}>
                <td className="mono">{d.event_id}</td>
                <td className="mono">{d.error_reason ?? "—"}</td>
                <td className="mono">
                  {d.root_cause ?? "—"}
                  {d.diagnosis_tier === 2 && (
                    <span className="tag tier2" style={{ marginLeft: 6 }}>
                      tier 2
                    </span>
                  )}
                </td>
                <td className="mono">{d.intended_action}</td>
                <td className="mono">{d.executed_action}</td>
                <td className="num">{rupees(d.amount_paise)}</td>
                <td>{dispositionTag(d)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <div className="empty">no decisions match</div>}
      </div>

      {open && (
        <div className="drawer" onClick={() => setOpen(null)}>
          <div className="body" onClick={(e) => e.stopPropagation()}>
            <h3>{open.event_id}</h3>
            <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 0 }}>
              Every gate that ran, in the words it used at the time.
            </p>
            <pre className="trace">{open.explain}</pre>
            {Object.keys(open.metadata).length > 0 && (
              <>
                <h2 style={{ marginTop: 20, fontSize: 12, color: "var(--muted)" }}>EXECUTION</h2>
                <pre className="trace">
                  {Object.entries(open.metadata)
                    .map(([k, v]) => `${k.padEnd(20)} ${v}`)
                    .join("\n")}
                </pre>
              </>
            )}
            <p className="note mono" style={{ color: "var(--muted)", fontSize: 12 }}>
              hash {open.record_hash.slice(0, 32)}…
            </p>
          </div>
        </div>
      )}
    </>
  );
}

function Approvals({ reviewer, onChange }: { reviewer: string; onChange: () => void }) {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [total, setTotal] = useState(0);
  const [err, setErr] = useState("");

  const load = useCallback(() => {
    api.approvals().then((r) => {
      setItems(r.items);
      setTotal(r.total_paise);
    });
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const decide = async (eventId: string, approve: boolean) => {
    setErr("");
    try {
      await api.decideApproval(eventId, approve, reviewer, "");
      load();
      onChange();
    } catch (e) {
      setErr(String(e instanceof Error ? e.message : e));
    }
  };

  const pending = items.filter((i) => !i.review);

  return (
    <>
      <div className="cards">
        <div className="card warn">
          <div className="label">parked</div>
          <div className="value">{rupees(total)}</div>
          <div className="foot">{pending.length} awaiting a decision</div>
        </div>
      </div>
      {err && <p className="err">{err}</p>}

      <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
        <table>
          <thead>
            <tr>
              <th>event</th>
              <th>customer</th>
              <th style={{ textAlign: "right" }}>amount</th>
              <th>why parked</th>
              <th style={{ width: 210 }}>decision</th>
            </tr>
          </thead>
          <tbody>
            {items.map((i) => (
              <tr key={i.event_id}>
                <td className="mono">{i.event_id}</td>
                <td className="mono">{i.customer_id}</td>
                <td className="num">{rupees(i.amount_paise)}</td>
                <td>{i.reason}</td>
                <td>
                  {i.review ? (
                    <span className={i.review.approved === "yes" ? "tag allow" : "tag deny"}>
                      {i.review.approved === "yes" ? "approved" : "rejected"} · {i.review.reviewer}
                    </span>
                  ) : (
                    <span style={{ display: "flex", gap: 8 }}>
                      <button
                        className="primary"
                        disabled={!reviewer}
                        onClick={() => decide(i.event_id, true)}
                      >
                        Approve
                      </button>
                      <button
                        className="danger"
                        disabled={!reviewer}
                        onClick={() => decide(i.event_id, false)}
                      >
                        Reject
                      </button>
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!items.length && <div className="empty">nothing awaiting approval</div>}
      </div>
      <p className="note" style={{ color: "var(--muted)", fontSize: 12.5 }}>
        A decision here appends a new ledger record naming you. It never edits the original —
        that record described a moment when no human had looked, and that stays true of it.
      </p>
    </>
  );
}

function Rules({ reviewer, onChange }: { reviewer: string; onChange: () => void }) {
  const [items, setItems] = useState<PendingRule[]>([]);
  const [err, setErr] = useState("");
  const [done, setDone] = useState("");

  const load = useCallback(() => {
    api.rules().then((r) => setItems(r.items));
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const act = async (reason: string, approve: boolean) => {
    setErr("");
    setDone("");
    try {
      if (approve) {
        await api.approveRule(reason, reviewer, "");
        setDone(`${reason} promoted to tier 1 — resolved by table lookup from now on.`);
      } else {
        await api.rejectRule(reason, reviewer, "");
        setDone(`${reason} rejected.`);
      }
      load();
      onChange();
    } catch (e) {
      setErr(String(e instanceof Error ? e.message : e));
    }
  };

  return (
    <>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 0 }}>
        Codes tier 1 did not recognise, classified by tier 2. Tier 2 can never authorise
        contacting a customer — approving a rule here is what unlocks that, through the ordinary
        path, having been seen by a person.
      </p>
      {err && <p className="err">{err}</p>}
      {done && <p style={{ color: "var(--allow)", fontSize: 13 }}>{done}</p>}

      {items.map((r) => (
        <div className="rule" key={r.reason}>
          <div className="head">
            <span className="reason">{r.reason}</span>
            <span className="tag">{r.root_cause}</span>
            <span className="tag">retry {r.retry_class}</span>
            <span className="tag tier2">confidence {(r.confidence * 100).toFixed(0)}%</span>
            <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: 12.5 }}>
              seen {r.seen_count}× · {r.model}
            </span>
          </div>
          <p style={{ margin: "8px 0 0", color: "var(--muted)", fontSize: 13 }}>{r.rationale}</p>
          <div className="row">{r.taxonomy_row}</div>
          {r.would_unlock_contact && (
            <p className="warn">⚠ Approving this permits contacting customers about this code.</p>
          )}
          <div className="actions">
            <button className="primary" disabled={!reviewer} onClick={() => act(r.reason, true)}>
              Approve → tier 1
            </button>
            <button className="danger" disabled={!reviewer} onClick={() => act(r.reason, false)}>
              Reject
            </button>
          </div>
        </div>
      ))}
      {!items.length && <div className="empty">no rules awaiting review</div>}
    </>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [chain, setChain] = useState<Integrity | null>(null);
  const [reviewer, setReviewer] = useState(() => localStorage.getItem(REVIEWER_KEY) ?? "");
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    Promise.all([api.metrics(), api.integrity()])
      .then(([m, i]) => {
        setMetrics(m);
        setChain(i);
        setError("");
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)));
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    localStorage.setItem(REVIEWER_KEY, reviewer);
  }, [reviewer]);

  const tabs: [Tab, string, number | null | undefined][] = [
    ["overview", "Overview", null],
    ["decisions", "Decisions", metrics?.decisions],
    ["approvals", "Approvals", metrics?.approval_queue],
    ["rules", "Rules", metrics?.pending_rules],
  ];

  return (
    <div className="app">
      <header className="top">
        <h1>Recoup</h1>
        <span className="sub">recovery decisioning control plane</span>
        {chain && (
          <span className={chain.ok ? "chain ok" : "chain bad"}>
            {chain.ok
              ? `✓ chain verified · ${chain.records.toLocaleString()} records`
              : `✗ CHAIN BROKEN · ${chain.error}`}
          </span>
        )}
      </header>

      <div className="controls" style={{ marginTop: 14 }}>
        <input
          placeholder="your name, to sign reviews"
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
          style={{ width: 260 }}
        />
        <button onClick={refresh}>Refresh</button>
        {!reviewer && (
          <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
            Approvals are attributed, so a name is required before you can act.
          </span>
        )}
      </div>

      {error && (
        <p className="err">
          API unreachable — is python -m recoup.console.server running? ({error})
        </p>
      )}

      <nav className="tabs">
        {tabs.map(([id, label, badge]) => (
          <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            {label}
            {badge != null && badge > 0 && <span className="badge">{badge}</span>}
          </button>
        ))}
      </nav>

      {tab === "overview" && metrics && <Overview m={metrics} />}
      {tab === "decisions" && <Decisions />}
      {tab === "approvals" && <Approvals reviewer={reviewer} onChange={refresh} />}
      {tab === "rules" && <Rules reviewer={reviewer} onChange={refresh} />}
    </div>
  );
}
