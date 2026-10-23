import { useEffect, useMemo, useState } from "react";
import {
  Aggregate, api, CriterionInput, CriterionType, Example, Health, LABEL_TONE, Mode, post,
  ScreenResult, Sentence, streamScreen,
} from "../api";

interface Trial {
  nct_id: string;
  title: string;
  inclusion: string[];
  exclusion: string[];
}

export default function ScreenPage({ health }: { health: Health | null }) {
  const [examples, setExamples] = useState<Example[]>([]);
  const [note, setNote] = useState("");
  const [criteria, setCriteria] = useState<CriterionInput[]>([]);
  const [trialId, setTrialId] = useState("");
  const [trialTitle, setTrialTitle] = useState<string | null>(null);
  const [incText, setIncText] = useState("");
  const [excText, setExcText] = useState("");
  const [mode, setMode] = useState<Mode>("cached");

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const [results, setResults] = useState<(ScreenResult | undefined)[]>([]);
  const [aggregate, setAggregate] = useState<Aggregate | null>(null);
  const [reportId, setReportId] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [screenedCriteria, setScreenedCriteria] = useState<CriterionInput[]>([]);

  useEffect(() => {
    api<Example[]>("/api/examples").then(setExamples).catch(() => setExamples([]));
  }, []);

  const ready = health?.state === "ready";
  const done = results.filter(Boolean).length;

  function loadExample(index: number) {
    const ex = examples[index];
    if (!ex) return;
    setNote(ex.note);
    setCriteria(ex.criteria);
    setTrialId(ex.trial_id);
    setTrialTitle(`Validation example · patient ${ex.patient_id}`);
    setInfo(`Loaded ${ex.criteria.length} criteria for ${ex.patient_id} / ${ex.trial_id} from the validation split (expert labels shown for comparison).`);
    setError(null);
  }

  async function fetchTrial() {
    setError(null);
    setInfo(null);
    try {
      const t = await api<Trial>(`/api/trial/${encodeURIComponent(trialId.trim())}`);
      setTrialTitle(t.title);
      const items: CriterionInput[] = [
        ...t.inclusion.map((text) => ({ type: "inclusion" as const, text })),
        ...t.exclusion.map((text) => ({ type: "exclusion" as const, text })),
      ];
      if (!items.length) throw new Error("This trial lists no eligibility criteria.");
      setCriteria(items);
      setInfo(`Fetched ${t.inclusion.length} inclusion and ${t.exclusion.length} exclusion criteria from ClinicalTrials.gov.`);
    } catch (e) {
      setError(`Trial lookup failed: ${(e as Error).message}`);
    }
  }

  async function addFromText() {
    setError(null);
    try {
      const [inc, exc] = await Promise.all([
        post<{ items: string[] }>("/api/criteria/parse", { text: incText }),
        post<{ items: string[] }>("/api/criteria/parse", { text: excText }),
      ]);
      const items: CriterionInput[] = [
        ...inc.items.map((text) => ({ type: "inclusion" as const, text })),
        ...exc.items.map((text) => ({ type: "exclusion" as const, text })),
      ];
      if (!items.length) throw new Error("No criteria found in the pasted text.");
      setCriteria((c) => [...c, ...items]);
      setIncText("");
      setExcText("");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function updateCriterion(i: number, patch: Partial<CriterionInput>) {
    setCriteria((c) => c.map((x, j) => (j === i ? { ...x, ...patch, expert_label: undefined } : x)));
  }

  async function run() {
    setError(null);
    setInfo(null);
    if (note.trim().length < 10) return setError("Enter a patient note (at least a sentence).");
    const cleaned = criteria.filter((c) => c.text.trim().length > 0);
    if (!cleaned.length) return setError("Add at least one inclusion or exclusion criterion.");
    setRunning(true);
    setResults(new Array(cleaned.length).fill(undefined));
    setAggregate(null);
    setReportId(null);
    setSelected(null);
    setScreenedCriteria(cleaned);
    try {
      await streamScreen(
        {
          note,
          mode,
          trial_id: trialId.trim() || null,
          trial_title: trialTitle,
          criteria: cleaned.map(({ type, text }) => ({ type, text: text.trim() })),
        },
        (e) => {
          if (e.event === "start") setSentences(e.sentences);
          if (e.event === "result")
            setResults((r) => r.map((x, i) => (i === e.index ? e.result : x)));
          if (e.event === "done") {
            setAggregate(e.aggregate);
            setReportId(e.report_id);
          }
        },
      );
    } catch (e) {
      setError((e as Error).message);
      setResults([]);
    } finally {
      setRunning(false);
    }
  }

  const highlighted = useMemo(() => {
    const r = selected !== null ? results[selected] : undefined;
    return new Set(r?.evidence.map((e) => e.id) ?? []);
  }, [selected, results]);

  return (
    <div className="grid two">
      <section className="panel">
        <h2>1 · Patient note</h2>
        {examples.length > 0 && (
          <select className="full" defaultValue="" onChange={(e) => loadExample(Number(e.target.value))}>
            <option value="" disabled>Load a validation-split example…</option>
            {examples.map((ex, i) => (
              <option key={i} value={i}>
                {ex.patient_id} · {ex.trial_id} · {ex.criteria.length} criteria
              </option>
            ))}
          </select>
        )}
        <textarea
          className="note-input"
          placeholder="Paste or type a clinical note. Sentences are numbered automatically so evidence can be cited by ID."
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={9}
        />

        <h2>2 · Trial eligibility criteria</h2>
        <div className="row">
          <input
            placeholder="NCT ID, e.g. NCT01397994"
            value={trialId}
            onChange={(e) => setTrialId(e.target.value)}
          />
          <button className="secondary" onClick={fetchTrial} disabled={!trialId.trim()}>
            Fetch from ClinicalTrials.gov
          </button>
        </div>
        {trialTitle && <p className="hint">{trialTitle}</p>}

        <details className="paste">
          <summary>Paste criteria as text</summary>
          <label>Inclusion criteria (one per line or bulleted)</label>
          <textarea rows={4} value={incText} onChange={(e) => setIncText(e.target.value)} />
          <label>Exclusion criteria</label>
          <textarea rows={4} value={excText} onChange={(e) => setExcText(e.target.value)} />
          <button className="secondary" onClick={addFromText} disabled={!incText.trim() && !excText.trim()}>
            Add to list
          </button>
        </details>

        <div className="criteria-list">
          {criteria.map((c, i) => (
            <div key={i} className="criterion-edit">
              <select value={c.type} onChange={(e) => updateCriterion(i, { type: e.target.value as CriterionType })}>
                <option value="inclusion">Inclusion</option>
                <option value="exclusion">Exclusion</option>
              </select>
              <input value={c.text} onChange={(e) => updateCriterion(i, { text: e.target.value })} />
              <button className="icon" title="Remove" onClick={() => setCriteria((x) => x.filter((_, j) => j !== i))}>
                ✕
              </button>
            </div>
          ))}
          <button className="link" onClick={() => setCriteria((c) => [...c, { type: "inclusion", text: "" }])}>
            + Add criterion
          </button>
        </div>

        <h2>3 · Run</h2>
        <div className="row">
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            {Object.entries(health?.modes ?? { cached: "KV cache + prefix reuse (default)" }).map(([m, d]) => (
              <option key={m} value={m}>{d}</option>
            ))}
          </select>
          <button className="primary" onClick={run} disabled={running || !ready}>
            {running ? `Screening ${done}/${results.length}…` : `Screen ${criteria.length || ""} criteria`}
          </button>
        </div>
        {!ready && <p className="hint">The model is not ready yet; the button enables when it is.</p>}
        {error && <div className="alert bad">{error}</div>}
        {info && <div className="alert info">{info}</div>}
      </section>

      <section className="panel">
        {results.length === 0 ? (
          <div className="empty">
            <h2>Results</h2>
            <p>Load an example or enter a note and criteria, then run the screening. Each criterion is judged
              separately; click a result to highlight its evidence in the note.</p>
          </div>
        ) : (
          <>
            <Verdict aggregate={aggregate} running={running} done={done} total={results.length} />
            {reportId && (
              <div className="row export">
                <a className="button secondary" href={`/api/reports/${reportId}/download?fmt=html`}>Download report (HTML)</a>
                <a className="button secondary" href={`/api/reports/${reportId}/download?fmt=json`}>Download JSON</a>
                <span className="hint">Saved as report {reportId}</span>
              </div>
            )}

            <h3>Patient note · numbered sentences</h3>
            <ol className="note-view" start={0}>
              {sentences.map(([id, text]) => (
                <li key={id} value={id} className={highlighted.has(id) ? "hl" : ""}>{text}</li>
              ))}
            </ol>

            <h3>Criterion-level results</h3>
            {results.map((r, i) => (
              <ResultCard
                key={i}
                index={i}
                result={r}
                input={screenedCriteria[i]}
                selected={selected === i}
                onSelect={() => setSelected(selected === i ? null : i)}
              />
            ))}
          </>
        )}
      </section>
    </div>
  );
}

function Verdict({ aggregate, running, done, total }: { aggregate: Aggregate | null; running: boolean; done: number; total: number }) {
  if (!aggregate)
    return (
      <div className="verdict muted">
        <div className="progress"><div style={{ width: `${(100 * done) / Math.max(total, 1)}%` }} /></div>
        {running ? `Screening criterion ${Math.min(done + 1, total)} of ${total}…` : "Stopped"}
      </div>
    );
  const tone = aggregate.verdict === "likely ineligible" ? "bad" : aggregate.verdict === "potentially eligible" ? "good" : "warn";
  return (
    <div className={`verdict ${tone}`}>
      <div className="verdict-label">Preliminary outcome: <b>{aggregate.verdict}</b></div>
      <div className="hint">
        Rule-based aggregation of criterion predictions (not a ground-truth label):
        {aggregate.blocking_criteria.length > 0 && ` ${aggregate.blocking_criteria.length} blocking criterion(s);`}
        {aggregate.unresolved_criteria.length > 0 && ` ${aggregate.unresolved_criteria.length} need(s) more information;`}
        {(aggregate.absence_based_criteria?.length ?? 0) > 0 &&
          ` ${aggregate.absence_based_criteria!.length} 'not excluded' judgement(s) inferred only from absence — verify;`}
        {" "}see About for the exact rules.
      </div>
    </div>
  );
}

function ResultCard({ index, result, input, selected, onSelect }: {
  index: number; result?: ScreenResult; input?: CriterionInput; selected: boolean; onSelect: () => void;
}) {
  if (!result)
    return (
      <div className="card pending">
        <span className={`type ${input?.type}`}>{input?.type}</span> {input?.text}
        <div className="hint">waiting…</div>
      </div>
    );
  const tone = LABEL_TONE[result.label ?? ""] ?? "bad";
  const expertMatch = input?.expert_label ? input.expert_label === result.label : null;
  return (
    <div className={`card ${selected ? "selected" : ""}`} onClick={onSelect}>
      <div className="card-head">
        <span className="idx">#{index + 1}</span>
        <span className={`type ${result.criterion_type}`}>{result.criterion_type}</span>
        <span className={`label ${tone}`}>{result.label ?? "unparseable"}</span>
        {result.flags.map((f) => <span key={f} className="flag">{f.replace(/_/g, " ")}</span>)}
      </div>
      <div className="criterion-text">{result.criterion_text}</div>
      {input?.expert_label && (
        <div className={`expert ${expertMatch ? "good" : "bad"}`}>
          Expert label: {input.expert_label} {expertMatch ? "✓" : "✗"}
          {input.expert_evidence && ` · expert evidence: [${input.expert_evidence.join(", ")}]`}
        </div>
      )}
      <div className="evidence">
        {result.evidence.length ? (
          result.evidence.map((e) => <div key={e.id}><b>[{e.id}]</b> {e.text}</div>)
        ) : (
          <div className="hint">No sentences cited.</div>
        )}
      </div>
      <div className="explanation">{result.explanation}</div>
      <div className="timing">
        {result.timing.n_tokens} tokens · {result.timing.total_s?.toFixed(2)} s
        {result.timing.reused_prefix_tokens ? ` · reused ${result.timing.reused_prefix_tokens}/${result.timing.prompt_tokens} prompt tokens from cache` : ""}
        {result.timing.acceptance_rate !== undefined ? ` · draft acceptance ${(100 * result.timing.acceptance_rate).toFixed(0)}%` : ""}
        <details onClick={(e) => e.stopPropagation()}><summary>raw output</summary><code>{result.raw_output}</code></details>
      </div>
    </div>
  );
}
