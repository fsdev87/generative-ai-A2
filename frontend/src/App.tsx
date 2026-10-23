import { useEffect, useState } from "react";
import { api, Health } from "./api";
import ScreenPage from "./pages/ScreenPage";
import BenchmarkPage from "./pages/BenchmarkPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import ReportsPage from "./pages/ReportsPage";
import AboutPage from "./pages/AboutPage";

const TABS = [
  ["screen", "Screen patient"],
  ["benchmark", "Inference benchmark"],
  ["experiments", "Experiments"],
  ["reports", "Reports"],
  ["about", "About"],
] as const;
type Tab = (typeof TABS)[number][0];

export default function App() {
  const [tab, setTab] = useState<Tab>("screen");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Poll the model status until it is ready (or failed), then stop.
  useEffect(() => {
    let timer: number | undefined;
    const poll = async () => {
      try {
        const h = await api<Health>("/api/health");
        setHealth(h);
        setHealthError(null);
        if (h.state === "loading") timer = window.setTimeout(poll, 3000);
      } catch (e) {
        setHealthError((e as Error).message);
        timer = window.setTimeout(poll, 5000);
      }
    };
    poll();
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">⚕</span>
          <div>
            <div className="title">Trial Criterion Screening</div>
            <div className="subtitle">Fine-tuned Qwen2.5 · evidence-grounded eligibility</div>
          </div>
        </div>
        <ModelStatus health={health} error={healthError} />
      </header>

      <div className="disclaimer" role="note">
        <b>Educational screening prototype.</b> Not a clinical enrollment decision — every result must be
        reviewed by qualified clinical staff.
      </div>

      <nav className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? "tab active" : "tab"} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>

      <main>
        {/* Pages stay mounted so inputs and results survive tab switches. */}
        <div hidden={tab !== "screen"}><ScreenPage health={health} /></div>
        <div hidden={tab !== "benchmark"}><BenchmarkPage health={health} /></div>
        {tab === "experiments" && <ExperimentsPage />}
        {tab === "reports" && <ReportsPage />}
        {tab === "about" && <AboutPage health={health} />}
      </main>
    </div>
  );
}

function ModelStatus({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <div className="status bad">● Server unreachable</div>;
  if (!health) return <div className="status muted">● Connecting…</div>;
  if (health.state === "loading") return <div className="status warn">● Loading model…</div>;
  if (health.state === "error") return <div className="status bad" title={health.error ?? ""}>● Model failed to load</div>;
  return (
    <div className="status good" title={`${health.target_model} + ${health.adapter ?? "no adapter"}`}>
      ● Ready · {health.target_model.split("/").pop()} {health.adapter ? "+ LoRA" : "(base)"} · {health.device}
    </div>
  );
}
