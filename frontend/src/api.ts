export type CriterionType = "inclusion" | "exclusion";
export type Mode = "cached" | "cached_tiled" | "uncached" | "speculative";
export type Sentence = [number, string];

export interface Health {
  state: "loading" | "ready" | "error";
  error: string | null;
  target_model: string;
  adapter: string | null;
  draft_model: string;
  dtype: string;
  device: string;
  max_prompt_tokens: number;
  max_new_tokens: number;
  spec_k: number;
  modes: Record<Mode, string>;
  draft_loaded: boolean;
  load_seconds?: number;
}

export interface CriterionInput {
  type: CriterionType;
  text: string;
  expert_label?: string;
  expert_evidence?: number[];
}

export interface Timing {
  n_tokens?: number;
  prefill_s?: number;
  decode_s?: number;
  total_s?: number;
  tokens_per_s?: number;
  reused_prefix_tokens?: number;
  prompt_tokens?: number;
  acceptance_rate?: number;
}

export interface ScreenResult {
  criterion_type: CriterionType;
  criterion_text: string;
  label: string | null;
  evidence: { id: number; text: string }[];
  explanation: string;
  flags: string[];
  raw_output: string;
  mode: Mode;
  timing: Timing;
}

export interface Aggregate {
  verdict: string;
  blocking_criteria: number[];
  unresolved_criteria: number[];
  counts: Record<string, Record<string, number>>;
  rules: string[];
}

export interface Example {
  patient_id: string;
  trial_id: string;
  split: string;
  note: string;
  criteria: CriterionInput[];
}

export interface BenchRow {
  mode: Mode;
  description: string;
  n_tokens: number;
  prefill_s: number;
  decode_s: number;
  total_s: number;
  tokens_per_s: number;
  decode_tokens_per_s: number;
  peak_memory_mb: number | null;
  cache_mb: number | null;
  tokens_equal_first_mode: boolean;
  text: string;
  acceptance_rate?: number;
  drafted?: number;
  accepted?: number;
  iterations?: number;
  mean_accepted_per_iteration?: number;
}

export interface BenchResult {
  prompt_tokens: number;
  max_new_tokens: number;
  rows: BenchRow[];
}

export interface AttentionResult {
  shape: number[];
  causal: boolean;
  block: number;
  score_matrix_mb: number;
  rows: { impl: string; median_ms: number; max_abs_err: number; mean_abs_err: number }[];
}

export interface ReportSummary {
  id: string;
  created_at: string;
  trial_id: string | null;
  verdict: string;
  n_criteria: number;
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI/pydantic validation errors: [{loc: [...], msg: "..."}]
    return detail
      .map((d: { loc?: (string | number)[]; msg?: string }) => {
        const where = (d.loc ?? []).filter((p) => p !== "body").join(" → ");
        return where ? `${where}: ${d.msg}` : d.msg ?? "invalid input";
      })
      .join("; ");
  }
  return "Request failed";
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  } catch {
    throw new Error("Cannot reach the server. Is the backend running?");
  }
  if (!resp.ok) {
    let message = `${resp.status} ${resp.statusText}`;
    try {
      message = formatDetail((await resp.json()).detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return resp.json() as Promise<T>;
}

export const post = <T,>(path: string, body: unknown) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body) });

export type ScreenEvent =
  | { event: "start"; sentences: Sentence[]; total: number }
  | { event: "result"; index: number; result: ScreenResult }
  | { event: "done"; report_id: string; aggregate: Aggregate };

/** POST /api/screen and call onEvent for every NDJSON line as it arrives. */
export async function streamScreen(body: unknown, onEvent: (e: ScreenEvent) => void) {
  let resp: Response;
  try {
    resp = await fetch("/api/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Cannot reach the server. Is the backend running?");
  }
  if (!resp.ok || !resp.body) {
    let message = `${resp.status} ${resp.statusText}`;
    try {
      message = formatDetail((await resp.json()).detail);
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line) as ScreenEvent);
    }
  }
}

export const LABEL_TONE: Record<string, string> = {
  included: "good",
  "not excluded": "good",
  "not included": "bad",
  excluded: "bad",
  "not enough information": "warn",
  "not applicable": "muted",
};
