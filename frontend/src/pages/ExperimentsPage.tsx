import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Json = any;

const COLORS = ["#2563eb", "#059669", "#c2410c", "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#a16207"];
// Before/after rows shown by default; everything else behind "show all runs".
const KEY_EVALS = ["eval/zeroshot_val_label_first", "eval/final_val", "eval/zeroshot_test", "eval/final_test"];

const runName = (path: string) => path.split("/").slice(0, -1).join("/");

export default function ExperimentsPage() {
  const [files, setFiles] = useState<Record<string, Json> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Record<string, Json>>("/api/results").then(setFiles).catch((e) => setError((e as Error).message));
  }, []);

  if (error) return <div className="alert bad">{error}</div>;
  if (!files) return <p>Loading experiment results…</p>;
  if (!Object.keys(files).length)
    return <div className="panel"><p>No experiment results found in <code>results/</code> yet.</p></div>;

  const pick = (suffix: string) => Object.entries(files).filter(([k]) => k.endsWith(suffix));
  return (
    <div className="stack">
      <p className="hint">
        Everything on this page is rendered from the JSON logs in <code>results/</code> produced by the
        training, evaluation and benchmark scripts (Kaggle T4 GPU unless stated otherwise).
      </p>
      <EvalComparison entries={pick("metrics.json")} />
      <ConfusionMatrices entries={pick("metrics.json")} />
      <TrainingCurves histories={pick("history.json")} />
      <EpochTable runs={pick("val_by_epoch.json")} />
      {Object.entries(files).filter(([k, v]) => k.includes("attention") && v.performance).map(([k, v]) =>
        <AttentionBench key={k} name={k} data={v} />)}
      {Object.entries(files).filter(([k, v]) => k.includes("decoding_gpu") && v.lengths).map(([k, v]) =>
        <DecodingBench key={k} name={k} data={v} />)}
    </div>
  );
}

function TrainingCurves({ histories }: { histories: [string, Json[]][] }) {
  const names = histories.map(([k]) => runName(k).replace("training/", ""));
  const [shown, setShown] = useState<string[]>(() =>
    names.filter((n) => ["lora_r16", "label_first", "oversample", "epochs5"].includes(n)));
  if (!histories.length) return null;
  return (
    <section className="panel">
      <h2>Training loss <span className="hint">answer tokens only, 15-step moving average</span></h2>
      <div className="row">
        {names.map((n, i) => (
          <label key={n} className="check">
            <input type="checkbox" checked={shown.includes(n)}
              onChange={() => setShown((s) => (s.includes(n) ? s.filter((x) => x !== n) : [...s, n]))} />
            <span style={{ color: COLORS[i % COLORS.length] }}>■</span> {n}
          </label>
        ))}
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" type="number" domain={[0, "dataMax"]} allowDuplicatedCategory={false}
              ticks={[0, 1, 2, 3, 4, 5]}
              label={{ value: "epoch", position: "insideBottom", offset: -4 }} />
            <YAxis scale="log" domain={["auto", "auto"]} allowDataOverflow
              label={{ value: "loss (log)", angle: -90, position: "insideLeft" }} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} labelFormatter={(e) => `epoch ${Number(e).toFixed(2)}`} />
            {histories.map(([k, h], i) => {
              const n = names[i];
              if (!shown.includes(n)) return null;
              // Per-step losses of micro-batches are noisy; show a 15-step moving average.
              const w = 15;
              const data = h.map((x: Json, j: number) => {
                const win = h.slice(Math.max(0, j - w + 1), j + 1);
                return { epoch: x.epoch, loss: win.reduce((s: number, y: Json) => s + y.loss, 0) / win.length };
              });
              return <Line key={k} data={data} dataKey="loss" name={n} stroke={COLORS[i % COLORS.length]}
                dot={false} strokeWidth={1.2} isAnimationActive={false} />;
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function EpochTable({ runs }: { runs: [string, Json[]][] }) {
  const [sel, setSel] = useState(() => runs.find(([k]) => k.includes("label_first/"))?.[0] ?? runs[0]?.[0]);
  if (!runs.length) return null;
  const rows = Object.fromEntries(runs)[sel] ?? [];
  return (
    <section className="panel">
      <h2>Validation after each epoch</h2>
      <select value={sel} onChange={(e) => setSel(e.target.value)}>
        {runs.map(([k]) => <option key={k} value={k}>{runName(k).replace("training/", "")}</option>)}
      </select>
      <table className="table">
        <thead><tr><th>Epoch</th><th>Val loss</th><th>Accuracy</th><th>Macro-F1</th><th>Evidence micro-F1</th><th>Valid JSON</th></tr></thead>
        <tbody>
          {rows.map((r: Json) => (
            <tr key={r.epoch}><td>{r.epoch}</td><td>{r.val_loss}</td><td>{r.accuracy}</td><td><b>{r.macro_f1}</b></td>
              <td>{r.evidence_micro_f1}</td><td>{(100 * r.parse_ok_rate).toFixed(0)}%</td></tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function EvalComparison({ entries }: { entries: [string, Json][] }) {
  const [all, setAll] = useState(false);
  if (!entries.length) return null;
  const rows = entries
    .filter(([k]) => all || KEY_EVALS.includes(runName(k)))
    .sort(([a], [b]) => {
      const ia = KEY_EVALS.indexOf(runName(a)), ib = KEY_EVALS.indexOf(runName(b));
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
  return (
    <section className="panel">
      <h2>Model evaluations: before vs. after fine-tuning</h2>
      <label className="check">
        <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
        show all runs (ablations, seeds, draft model)
      </label>
      <table className="table">
        <thead>
          <tr><th>Run</th><th>Subset</th><th>n</th><th>Accuracy</th><th>Macro-F1</th>
            <th>Evidence micro P / R / F1</th><th>Empty-gold agreement</th></tr>
        </thead>
        <tbody>
          {rows.flatMap(([k, m]) => ["overall", "inclusion", "exclusion"].filter((s) => m[s]).map((s) => {
            const c = m[s].classification, ev = m[s].evidence;
            return (
              <tr key={k + s} className={s === "overall" ? "" : "sub"}>
                <td>{s === "overall" ? runName(k).replace("eval/", "") : ""}</td><td>{s}</td><td>{c.n}</td>
                <td>{c.accuracy}</td><td><b>{c.macro_f1}</b></td>
                <td>{ev.micro.precision} / {ev.micro.recall} / {ev.micro.f1}</td>
                <td>{ev.empty_gold.agreement}</td>
              </tr>
            );
          }))}
        </tbody>
      </table>
    </section>
  );
}

function ConfusionMatrices({ entries }: { entries: [string, Json][] }) {
  const [sel, setSel] = useState(() => entries.find(([k]) => k === "eval/final_test/metrics.json")?.[0]
    ?? entries[0]?.[0]);
  const [subset, setSubset] = useState("exclusion");
  if (!entries.length) return null;
  const cls = Object.fromEntries(entries)[sel]?.[subset]?.classification;
  return (
    <section className="panel">
      <h2>Confusion matrix and per-class metrics</h2>
      <div className="row">
        <select value={sel} onChange={(e) => setSel(e.target.value)}>
          {entries.map(([k]) => <option key={k} value={k}>{runName(k).replace("eval/", "")}</option>)}
        </select>
        <select value={subset} onChange={(e) => setSubset(e.target.value)}>
          {["overall", "inclusion", "exclusion"].map((s) => <option key={s}>{s}</option>)}
        </select>
      </div>
      {cls && (
        <div className="grid two">
          <table className="table cm">
            <thead><tr><th>gold ↓ / predicted →</th>{cls.confusion_matrix.labels.map((l: string) => <th key={l}>{l}</th>)}</tr></thead>
            <tbody>
              {cls.confusion_matrix.matrix.map((row: number[], i: number) => {
                const max = Math.max(1, ...cls.confusion_matrix.matrix.flat());
                return (
                  <tr key={i}>
                    <th>{cls.confusion_matrix.labels[i]}</th>
                    {row.map((v, j) => (
                      <td key={j} style={{ background: `rgba(37,99,235,${(0.06 + (0.75 * v) / max).toFixed(2)})`,
                        color: v / max > 0.6 ? "#fff" : undefined, fontWeight: i === j ? 700 : 400 }}>{v}</td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="table">
            <thead><tr><th>Class</th><th>P</th><th>R</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>
              {Object.entries(cls.per_class).map(([l, v]: [string, Json]) => (
                <tr key={l}><td>{l}</td><td>{v.precision}</td><td>{v.recall}</td><td>{v.f1}</td><td>{v.support}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function AttentionBench({ name, data }: { name: string; data: Json }) {
  const impls = useMemo(() => [...new Set(data.performance.map((r: Json) => r.impl))] as string[], [data]);
  const series = (key: string) => impls.map((impl) => ({
    impl, points: data.performance.filter((r: Json) => r.impl === impl && r[key] !== undefined)
      .map((r: Json) => ({ seq_len: r.seq_len, v: r[key] })),
  }));
  const chart = (key: string, ylabel: string) => (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="seq_len" type="number" scale="log" domain={["dataMin", "dataMax"]}
          ticks={[...new Set(data.performance.map((r: Json) => r.seq_len))] as number[]} allowDuplicatedCategory={false}
          label={{ value: "sequence length", position: "insideBottom", offset: -4 }} />
        <YAxis scale="log" domain={["auto", "auto"]} label={{ value: ylabel, angle: -90, position: "insideLeft" }} />
        <Tooltip />
        <Legend />
        {series(key).map((s, i) => (
          <Line key={s.impl} data={s.points} dataKey="v" name={s.impl} stroke={COLORS[i % COLORS.length]}
            isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
  const hasMem = data.performance.some((r: Json) => r.peak_extra_mb !== undefined);
  return (
    <section className="panel">
      <h2>Attention kernels <span className="hint">{name} · {data.device} · {data.performance_dtype}</span></h2>
      <div className={hasMem ? "grid two" : ""}>
        <div className="chart">{chart("median_ms", "median ms (log)")}</div>
        {hasMem && <div className="chart">{chart("peak_extra_mb", "peak extra MB (log)")}</div>}
      </div>
    </section>
  );
}

function DecodingBench({ name, data }: { name: string; data: Json }) {
  const modes = Object.keys(data.lengths[0]?.modes ?? {});
  return (
    <section className="panel">
      <h2>Decoding benchmark <span className="hint">{name} · {data.device} · {data.dtype}</span></h2>
      <p className="hint">Total seconds per configuration (median of {data.repeats}); ✗ marks a token mismatch vs. KV-cached greedy.</p>
      <table className="table">
        <thead>
          <tr><th>Prompt</th><th>New</th>{modes.map((m) => <th key={m}>{m}</th>)}</tr>
        </thead>
        <tbody>
          {data.lengths.map((e: Json, i: number) => (
            <tr key={i}>
              <td>{e.prompt_tokens}</td><td>{e.new_tokens}</td>
              {modes.map((m) => {
                const r = e.modes[m];
                return <td key={m}>{r ? <>{r.total_s.toFixed(2)}{r.tokens_equal_cached ? "" : " ✗"}</> : "–"}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {data.task?.n_prompts && (
        <p>
          Screening prompts (n={data.task.n_prompts}, k={data.task.k}): identical outputs {data.task.tokens_equal}/{data.task.n_prompts},
          draft acceptance {(100 * data.task.acceptance_rate).toFixed(1)}%, speed-up ×{data.task.speedup}.
        </p>
      )}
    </section>
  );
}
