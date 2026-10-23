import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, AttentionResult, BenchResult, CriterionType, Example, Health, Mode, post } from "../api";

const MODE_COLORS: Record<Mode, string> = {
  uncached: "#c2410c",
  cached: "#2563eb",
  cached_tiled: "#7c3aed",
  speculative: "#059669",
};

interface HistoryPoint {
  run: number;
  max_new_tokens: number;
  prompt_tokens: number;
  [mode: string]: number;
}

export default function BenchmarkPage({ health }: { health: Health | null }) {
  const [note, setNote] = useState("");
  const [ctype, setCtype] = useState<CriterionType>("inclusion");
  const [ctext, setCtext] = useState("");
  const [modes, setModes] = useState<Mode[]>(["uncached", "cached", "speculative"]);
  const [maxNew, setMaxNew] = useState(48);
  const [stopAtEos, setStopAtEos] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);

  useEffect(() => {
    api<Example[]>("/api/examples?limit=1").then((ex) => {
      if (!ex.length) return;
      setNote(ex[0].note);
      setCtype(ex[0].criteria[0].type);
      setCtext(ex[0].criteria[0].text);
    }).catch(() => undefined);
  }, []);

  const ready = health?.state === "ready";
  const toggle = (m: Mode) => setModes((ms) => (ms.includes(m) ? ms.filter((x) => x !== m) : [...ms, m]));

  async function run() {
    setError(null);
    if (!modes.length) return setError("Select at least one decoding mode.");
    setRunning(true);
    try {
      // Cached first so equality is always judged against the KV-cached greedy output.
      const ordered = (["cached", "uncached", "cached_tiled", "speculative"] as Mode[]).filter((m) => modes.includes(m));
      const r = await post<BenchResult>("/api/benchmark", {
        note, criterion_type: ctype, criterion_text: ctext, modes: ordered,
        max_new_tokens: maxNew, stop_at_eos: stopAtEos,
      });
      setResult(r);
      setHistory((h) => [
        ...h,
        {
          run: h.length + 1, max_new_tokens: r.max_new_tokens, prompt_tokens: r.prompt_tokens,
          ...Object.fromEntries(r.rows.map((row) => [row.mode, row.total_s])),
        },
      ]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  const chartRows = result?.rows.map((r) => ({
    mode: r.mode, prefill: +r.prefill_s.toFixed(3), decode: +r.decode_s.toFixed(3), tps: r.tokens_per_s,
  }));

  return (
    <div className="grid two">
      <section className="panel">
        <h2>Decoding benchmark (live)</h2>
        <p className="hint">
          Runs the same prompt through each decoding implementation on this server, with greedy decoding and a
          fresh cache per mode. Equality is checked token-by-token against the KV-cached output.
        </p>
        <label>Patient note</label>
        <textarea rows={6} value={note} onChange={(e) => setNote(e.target.value)} />
        <div className="row">
          <select value={ctype} onChange={(e) => setCtype(e.target.value as CriterionType)}>
            <option value="inclusion">Inclusion</option>
            <option value="exclusion">Exclusion</option>
          </select>
          <input value={ctext} onChange={(e) => setCtext(e.target.value)} placeholder="Criterion text" />
        </div>
        <label>Modes</label>
        <div className="checks">
          {(Object.keys(MODE_COLORS) as Mode[]).map((m) => (
            <label key={m} className="check">
              <input type="checkbox" checked={modes.includes(m)} onChange={() => toggle(m)} />
              <span style={{ color: MODE_COLORS[m] }}>■</span> {health?.modes?.[m] ?? m}
            </label>
          ))}
        </div>
        <label>New tokens: {maxNew}</label>
        <input type="range" min={8} max={128} step={8} value={maxNew} onChange={(e) => setMaxNew(+e.target.value)} />
        <label className="check">
          <input type="checkbox" checked={stopAtEos} onChange={(e) => setStopAtEos(e.target.checked)} />
          Stop at end-of-answer token (otherwise generate exactly N tokens for a fair comparison)
        </label>
        <button className="primary" onClick={run} disabled={running || !ready}>
          {running ? "Running… (uncached is slow on CPU)" : "Run benchmark"}
        </button>
        {error && <div className="alert bad">{error}</div>}

        <AttentionCheck />
      </section>

      <section className="panel">
        {!result ? (
          <div className="empty"><h2>Results</h2><p>Run a benchmark to compare decoding modes.</p></div>
        ) : (
          <>
            <h3>Prompt {result.prompt_tokens} tokens · {result.max_new_tokens} new tokens</h3>
            <table className="table">
              <thead>
                <tr><th>Mode</th><th>Prefill s</th><th>Decode s</th><th>Total s</th><th>tok/s</th><th>Same tokens</th><th>Draft acceptance</th></tr>
              </thead>
              <tbody>
                {result.rows.map((r) => (
                  <tr key={r.mode}>
                    <td><span style={{ color: MODE_COLORS[r.mode] }}>■</span> {r.mode}</td>
                    <td>{r.prefill_s.toFixed(3)}</td>
                    <td>{r.decode_s.toFixed(3)}</td>
                    <td><b>{r.total_s.toFixed(3)}</b></td>
                    <td>{r.tokens_per_s.toFixed(1)}</td>
                    <td className={r.tokens_equal_first_mode ? "good-text" : "bad-text"}>{r.tokens_equal_first_mode ? "✓ identical" : "✗ differs"}</td>
                    <td>{r.acceptance_rate !== undefined
                      ? `${(100 * r.acceptance_rate).toFixed(0)}% (${r.accepted}/${r.drafted}, ${r.iterations} rounds)`
                      : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="chart">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="mode" />
                  <YAxis label={{ value: "seconds", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="prefill" stackId="t" fill="#94a3b8" name="prefill (s)" isAnimationActive={false} />
                  <Bar dataKey="decode" stackId="t" fill="#2563eb" name="decode (s)" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <details>
              <summary>Generated text per mode</summary>
              {result.rows.map((r) => <div key={r.mode} className="mono"><b>{r.mode}:</b> {r.text}</div>)}
            </details>
          </>
        )}
        {history.length > 1 && (
          <>
            <h3>This session: total time per run</h3>
            <div className="chart">
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="run" tickFormatter={(r) => `#${r}`} />
                  <YAxis label={{ value: "seconds", angle: -90, position: "insideLeft" }} />
                  <Tooltip labelFormatter={(r) => {
                    const h = history[Number(r) - 1];
                    return h ? `run ${r}: ${h.prompt_tokens} prompt / ${h.max_new_tokens} new tokens` : `run ${r}`;
                  }} />
                  <Legend />
                  {(Object.keys(MODE_COLORS) as Mode[]).map((m) => (
                    <Line key={m} dataKey={m} stroke={MODE_COLORS[m]} connectNulls dot isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function AttentionCheck() {
  const [seqLen, setSeqLen] = useState(512);
  const [block, setBlock] = useState(64);
  const [causal, setCausal] = useState(true);
  const [result, setResult] = useState<AttentionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await post<AttentionResult>("/api/attention-check", { seq_len: seqLen, block, causal }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="subpanel">
      <h3>Attention kernels (live)</h3>
      <p className="hint">Random q/k/v at Qwen2.5-1.5B's head shape (12 heads × 128). Errors are against a float64 reference.</p>
      <div className="row">
        <label>Seq len <input type="number" min={8} max={2048} value={seqLen} onChange={(e) => setSeqLen(+e.target.value)} /></label>
        <label>Block <input type="number" min={8} max={512} value={block} onChange={(e) => setBlock(+e.target.value)} /></label>
        <label className="check"><input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} /> causal</label>
        <button className="secondary" onClick={run} disabled={busy}>{busy ? "Running…" : "Compare"}</button>
      </div>
      {error && <div className="alert bad">{error}</div>}
      {result && (
        <table className="table">
          <thead><tr><th>Implementation</th><th>Median ms</th><th>Max |err|</th><th>Mean |err|</th></tr></thead>
          <tbody>
            {result.rows.map((r) => (
              <tr key={r.impl}>
                <td>{r.impl}</td><td>{r.median_ms.toFixed(2)}</td>
                <td>{r.max_abs_err.toExponential(2)}</td><td>{r.mean_abs_err.toExponential(2)}</td>
              </tr>
            ))}
          </tbody>
          <caption>Full score matrix would be {result.score_matrix_mb} MB; tiling keeps only {block}×{block} blocks.</caption>
        </table>
      )}
    </div>
  );
}
