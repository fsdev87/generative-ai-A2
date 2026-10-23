import { useEffect, useState } from "react";
import { api, ReportSummary } from "../api";

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<ReportSummary[]>("/api/reports").then(setReports).catch((e) => setError((e as Error).message));
  }, []);

  if (error) return <div className="alert bad">{error}</div>;
  if (!reports) return <p>Loading reports…</p>;
  return (
    <section className="panel">
      <h2>Saved screening reports</h2>
      {!reports.length ? (
        <p>No reports yet. Every completed screening is saved here automatically.</p>
      ) : (
        <table className="table">
          <thead><tr><th>Created</th><th>Report</th><th>Trial</th><th>Criteria</th><th>Preliminary outcome</th><th>Download</th></tr></thead>
          <tbody>
            {reports.map((r) => (
              <tr key={r.id}>
                <td>{r.created_at}</td><td className="mono">{r.id}</td><td>{r.trial_id ?? "–"}</td>
                <td>{r.n_criteria}</td><td>{r.verdict}</td>
                <td>
                  <a href={`/api/reports/${r.id}/download?fmt=html`}>HTML</a> ·{" "}
                  <a href={`/api/reports/${r.id}/download?fmt=json`}>JSON</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
