// Typed client for the console API. Types mirror the FastAPI responses; if they
// drift, that is a bug worth failing loudly on rather than papering over with any.

export interface Gate {
  gate: string;
  disposition: "allow" | "deny" | "needs_approval";
  reason: string;
}

export interface Decision {
  seq: number;
  decided_at: string;
  event_id: string;
  customer_id: string;
  amount_paise: number;
  error_reason: string | null;
  root_cause: string | null;
  diagnosis_tier: number | null;
  intended_action: string;
  executed_action: string;
  disposition: string;
  arm: string;
  recovered: boolean | null;
  denied_by: string[];
  gates: Gate[];
  metadata: Record<string, string>;
  record_hash: string;
  explain: string;
}

export interface Metrics {
  decisions: number;
  at_risk_paise: number;
  holdout: number;
  by_action: Record<string, number>;
  denials_by_gate: Record<string, number>;
  not_chased_paise: number;
  approval_queue: number;
  approval_queue_paise: number;
  ops_queue: number;
  ops_queue_paise: number;
  pending_rules: number;
  escalation_calls: number;
}

export interface QueueItem {
  event_id: string;
  customer_id: string;
  amount_paise: string;
  reason: string;
  review?: { reviewer: string; note: string; approved: string; at: string } | null;
}

export interface PendingRule {
  reason: string;
  root_cause: string;
  retry_class: string;
  new_instrument: boolean;
  customer_action: boolean;
  owner: string;
  in_scope: boolean;
  confidence: number;
  rationale: string;
  model: string;
  seen_count: number;
  taxonomy_row: string;
  would_unlock_contact: boolean;
}

export interface Integrity {
  ok: boolean;
  records: number;
  head?: string;
  error?: string;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${path}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  metrics: () => get<Metrics>("/api/metrics"),
  integrity: () => get<Integrity>("/api/integrity"),
  decisions: (params: { blocked_only?: boolean; tier?: number; limit?: number }) => {
    const q = new URLSearchParams();
    if (params.blocked_only) q.set("blocked_only", "true");
    if (params.tier != null) q.set("tier", String(params.tier));
    q.set("limit", String(params.limit ?? 100));
    return get<{ total: number; items: Decision[] }>(`/api/decisions?${q}`);
  },
  approvals: () => get<{ count: number; total_paise: number; items: QueueItem[] }>("/api/queues/approval"),
  ops: () => get<{ count: number; total_paise: number; items: QueueItem[] }>("/api/queues/ops"),
  rules: () => get<{ count: number; items: PendingRule[] }>("/api/rules/pending"),
  decideApproval: (eventId: string, approve: boolean, reviewer: string, note: string) =>
    post(`/api/queues/approval/${eventId}/decide?approve=${approve}`, { reviewer, note }),
  approveRule: (reason: string, reviewer: string, note: string) =>
    post(`/api/rules/${encodeURIComponent(reason)}/approve`, { reviewer, note }),
  rejectRule: (reason: string, reviewer: string, note: string) =>
    post(`/api/rules/${encodeURIComponent(reason)}/reject`, { reviewer, note }),
};

export const rupees = (paise: number | string): string =>
  `₹${(Number(paise) / 100).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
