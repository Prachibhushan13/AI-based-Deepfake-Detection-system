import { useEffect, useState } from "react";
import { StatCard } from "../components/ui/StatCard";
import { ConfidenceChart } from "../components/charts/ConfidenceChart";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

export function DashboardPage() {
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const loadHistory = async () => {
      try {
        const response = await api.get("/history");
        if (active) {
          setHistory(response.data);
        }
      } catch (error) {
        console.error("Failed to load dashboard history", error);
        if (active) {
          setHistory([]);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        loadHistory();
      }
    };

    loadHistory();
    const intervalId = window.setInterval(loadHistory, 10000);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      active = false;
      window.clearInterval(intervalId);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  const fakeCount = history.filter((item) => item.result === "FAKE").length;
  const latest = history[0];
  const chartData = latest?.frameTimeline ?? [
    { frameIndex: 0, score: 0.12 },
    { frameIndex: 1, score: 0.34 },
    { frameIndex: 2, score: 0.68 },
    { frameIndex: 3, score: 0.75 },
  ];

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="glass-panel rounded-[32px] p-8">
          <p className="text-sm uppercase tracking-[0.3em] text-cyber">Mission Control</p>
          <h1 className="mt-3 font-display text-4xl font-bold">Deepfake detection operations dashboard</h1>
          <p className="mt-3 max-w-2xl text-slate-300">
            Monitor detection volume, review forensic outputs, and track temporal confidence drift across analyzed videos.
          </p>
        </div>
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
          <StatCard label="Analyses" value={String(history.length)} hint="Videos processed in your workspace" />
          <StatCard label="Flagged Fake" value={String(fakeCount)} hint="Videos classified as manipulated" />
          <StatCard label="Real Videos" value={String(history.length - fakeCount)} hint="Authentic classifications" />
          <StatCard label="Latest Confidence" value={`${latest?.confidence ?? 0}%`} hint="Confidence score of most recent run" />
        </div>
        <ConfidenceChart data={chartData} />
        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="glass-panel rounded-[32px] p-6">
            <h2 className="font-display text-2xl font-semibold">Latest Detection Snapshot</h2>
            {latest ? (
              <div className="mt-5 space-y-3 text-slate-300">
                <p><span className="font-medium text-white">File:</span> {latest.filename}</p>
                <p><span className="font-medium text-white">Verdict:</span> {latest.result}</p>
                <p><span className="font-medium text-white">Confidence:</span> {latest.confidence}%</p>
                <p><span className="font-medium text-white">Model mode:</span> {latest.modelMode}</p>
                <p><span className="font-medium text-white">Captured:</span> {new Date(latest.createdAt).toLocaleString()}</p>
              </div>
            ) : (
              <p className="mt-5 text-slate-300">
                {loading ? "Loading analysis history..." : "No detections yet. Run an analysis and it will appear here automatically."}
              </p>
            )}
          </div>
          <div className="glass-panel rounded-[32px] p-6">
            <h2 className="font-display text-2xl font-semibold">Recent Analyses</h2>
            <div className="mt-5 space-y-3">
              {history.slice(0, 4).map((item) => (
                <div key={item.id} className="rounded-2xl border border-white/10 bg-slate-950/35 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="font-medium text-white">{item.filename}</p>
                      <p className="text-sm text-slate-400">{new Date(item.createdAt).toLocaleString()}</p>
                    </div>
                    <div className={`rounded-full px-3 py-1 text-xs ${item.result === "FAKE" ? "bg-alert/20 text-alert" : "bg-cyber/20 text-cyber"}`}>
                      {item.result}
                    </div>
                  </div>
                </div>
              ))}
              {!loading && history.length === 0 && (
                <p className="text-slate-300">Your analysis history is still empty. New prediction records will surface here after the backend stores them.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
