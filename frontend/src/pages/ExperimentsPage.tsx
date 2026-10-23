import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Json = any;

const COLORS = ["#2563eb", "#059669", "#c2410c", "#7c3aed", "#db2777", "#0891b2", "#65a30d"];

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
      {pick("history.json").map(([k, v]) => <TrainingCurve key={k} name={k} history={v} />)}
      {pick("val_by_epoch.json").map(([k, v]) => <EpochTable key={k} name={k} rows={v} />)}
      <EvalComparison entries={pick("metrics.json")} />
      {pick("metrics.json").length > 0 && <ConfusionMatrices entries={pick("metrics.json")} />}
      {Object.entries(files).filter(([k]) => k.includes("attention") && files[k].performance).map(([k, v]) =>
        <AttentionBench key={k} name={k} data={v} />)}
      {Object.entries(files).filter(([k]) => k.includes("decoding") && files[k].lengths).map(([k, v]) =>
        <DecodingBench key={k} name={k} data={v} />)}
    </div>
  );
}

function TrainingCurve({ name, history }: { name: string; history: Json[] }) {
  return (
    <section className="panel">
      <h2>Training loss <span className="hint">{name}</span></h2>
      <div className="chart">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" label={{ value: "optimizer step", position: "insideBottom", offset: -4 }} />
            <YAxis />
            <Tooltip />
            <Line dataKey="loss" stroke="#2563eb" dot={false} name="train loss (answer tokens)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function EpochTable({ name, rows }: { name: string; rows: Json[] }) {
  return (
    <section className="panel">
      <h2>Validation by epoch <span className="hint">{name}</span></h2>
      <table className="table">
        <thead><tr><th>Epoch</th><th>Val loss</th><th>Accuracy</th><th>Macro-F1</th><th>Evidence micro-F1</th><th>Valid JSON</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.epoch}><td>{r.epoch}</td><td>{r.val_loss}</td><td>{r.accuracy}</td><td><b>{r.macro_f1}</b></td>
              <td>{r.evidence_micro_f1}</td><td>{(100 * r.parse_ok_rate).toFixed(0)}%</td></tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function EvalComparison({ entries }: { entries: [string, Json][] }) {
  if (!entries.length) return null;
  const subsets = ["overall", "inclusion", "exclusion"];
  return (
    <section className="panel">
      <h2>Model evaluations (before / after fine-tuning)</h2>
      <table className="table">
        <thead>
          <tr><th>Run</th><th>Subset</th><th>n</th><th>Accuracy</th><th>Macro-F1</th><th>Evidence micro P / R / F1</th><th>Empty-gold agreement</th></tr>
        </thead>
        <tbody>
          {entries.flatMap(([k, m]) => subsets.filter((s) => m[s]).map((s) => {
            const c = m[s].classification, ev = m[s].evidence;
            return (
              <tr key={k + s}>
                <td>{s === "overall" ? k.replace("/metrics.json", "") : ""}</td><td>{s}</td><td>{c.n}</td>
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
  const [sel, setSel] = useState(entries[entries.length - 1][0]);
  const [subset, setSubset] = useState("overall");
  const metrics = Object.fromEntries(entries)[sel];
  const cls = metrics?.[subset]?.classification;
  return (
    <section className="panel">
      <h2>Confusion matrix & per-class metrics</h2>
      <div className="row">
        <select value={sel} onChange={(e) => setSel(e.target.value)}>
          {entries.map(([k]) => <option key={k} value={k}>{k.replace("/metrics.json", "")}</option>)}
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
                      <td key={j} style={{ background: `rgba(37,99,235,${(0.08 + (0.8 * v) / max).toFixed(2)})`, fontWeight: i === j ? 700 : 400 }}>{v}</td>
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
  const rows = useMemo(() => {
    const bySeq: Record<number, Json> = {};
    for (const r of data.performance) if (r.median_ms !== undefined) (bySeq[r.seq_len] ??= { seq_len: r.seq_len })[r.impl] = r.median_ms;
    return Object.values(bySeq);
  }, [data]);
  const impls = [...new Set(data.performance.map((r: Json) => r.impl))] as string[];
  return (
    <section className="panel">
      <h2>Attention runtime <span className="hint">{name} · {data.device} · {data.performance_dtype}</span></h2>
      <div className="chart">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="seq_len" label={{ value: "sequence length", position: "insideBottom", offset: -4 }} />
            <YAxis scale="log" domain={["auto", "auto"]} label={{ value: "ms (log)", angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Legend />
            {impls.map((m, i) => <Line key={m} dataKey={m} stroke={COLORS[i % COLORS.length]} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function DecodingBench({ name, data }: { name: string; data: Json }) {
  const modes = Object.keys(data.lengths[0]?.modes ?? {});
  return (
    <section className="panel">
      <h2>Decoding benchmark <span className="hint">{name} · {data.device} · {data.dtype}</span></h2>
      <table className="table">
        <thead>
          <tr><th>Prompt</th><th>New</th>{modes.map((m) => <th key={m}>{m} s</th>)}</tr>
        </thead>
        <tbody>
          {data.lengths.map((e: Json, i: number) => (
            <tr key={i}>
              <td>{e.prompt_tokens}</td><td>{e.new_tokens}</td>
              {modes.map((m) => {
                const r = e.modes[m];
                return <td key={m}>{r ? <>{r.total_s.toFixed(3)} {r.tokens_equal_cached ? "" : "✗"}</> : "–"}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {data.task?.n_prompts && (
        <p>
          Screening prompts (n={data.task.n_prompts}, k={data.task.k}): identical outputs {data.task.tokens_equal}/{data.task.n_prompts},
          acceptance {(100 * data.task.acceptance_rate).toFixed(1)}%, speed-up ×{data.task.speedup}.
        </p>
      )}
    </section>
  );
}
