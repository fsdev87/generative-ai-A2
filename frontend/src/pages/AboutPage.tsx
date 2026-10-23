import { Health } from "../api";

const RULES = [
  "Likely ineligible: any inclusion criterion is 'not included' or any exclusion criterion is 'excluded'.",
  "Needs clinician review: otherwise, if any criterion is 'not enough information' or could not be parsed.",
  "Potentially eligible: otherwise (every criterion is satisfied or 'not applicable').",
  "The verdict is a rule-based aggregation of criterion-level predictions, not a ground-truth label.",
];

export default function AboutPage({ health }: { health: Health | null }) {
  return (
    <div className="grid two">
      <section className="panel">
        <h2>What this is</h2>
        <p>
          An educational research prototype that judges one clinical-trial eligibility criterion at a time
          against a patient note, using a Qwen2.5-1.5B-Instruct model fine-tuned with LoRA on the
          TrialGPT Criterion Annotations dataset (patient-grouped 37/8/8 split, seed 42).
        </p>
        <p>
          For every criterion the model outputs one of the dataset's expert label categories and the IDs of the
          patient sentences that support it. Explanations are assembled only from those cited sentences, and a
          decisive label without cited evidence is flagged as <i>unsupported</i>. Missing information is never
          treated as proof of eligibility or ineligibility.
        </p>
        <h3>Labels</h3>
        <ul>
          <li><b>Inclusion:</b> included · not included · not enough information · not applicable</li>
          <li><b>Exclusion:</b> excluded · not excluded · not enough information · not applicable</li>
        </ul>
        <h3>Overall outcome rules</h3>
        <ol>{RULES.map((r) => <li key={r}>{r}</li>)}</ol>
        <div className="alert warn">
          Not a medical device and not a clinical enrollment decision. Results must be reviewed by qualified staff.
        </div>
      </section>
      <section className="panel">
        <h2>Model & inference settings</h2>
        {health ? (
          <table className="table">
            <tbody>
              <tr><th>Status</th><td>{health.state}{health.error ? ` – ${health.error}` : ""}</td></tr>
              <tr><th>Target model</th><td>{health.target_model}</td></tr>
              <tr><th>LoRA adapter</th><td>{health.adapter ?? "none (base model)"}</td></tr>
              <tr><th>Draft model</th><td>{health.draft_model} {health.draft_loaded ? "(loaded)" : "(loads on first speculative run)"}</td></tr>
              <tr><th>Device / dtype</th><td>{health.device} / {health.dtype}</td></tr>
              <tr><th>Max prompt tokens</th><td>{health.max_prompt_tokens} (longer inputs are rejected, never truncated)</td></tr>
              <tr><th>Max new tokens</th><td>{health.max_new_tokens}</td></tr>
              <tr><th>Speculative k</th><td>{health.spec_k}</td></tr>
              {health.load_seconds && <tr><th>Model load time</th><td>{health.load_seconds} s</td></tr>}
            </tbody>
          </table>
        ) : <p>Connecting…</p>}
        <h3>Decoding modes</h3>
        <ul>{health && Object.entries(health.modes).map(([m, d]) => <li key={m}><b>{m}</b>: {d}</li>)}</ul>
        <p className="hint">All modes use greedy decoding and our own Qwen2 forward pass with an explicit KV cache.</p>
      </section>
    </div>
  );
}
