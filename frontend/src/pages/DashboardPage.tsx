import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
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
    api
      .get<Prediction[]>("/history")
      .then((response) => {
        if (active) {
          setHistory(Array.isArray(response.data) ? response.data : []);
        }
      })
      .catch(() => {
        if (active) {
          setHistory([]);
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const fakeCount = history.filter((item) => item.result === "FAKE").length;
  const latest = history[0];

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
              Overview
            </p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight text-zinc-900">
              Session Summary
            </h1>
            <p className="mt-1 text-sm text-zinc-600">
              Summary of media analyses processed by the backend during your workspace sessions.
            </p>
          </div>
          <Link
            to="/"
            className="inline-flex items-center self-start rounded bg-zinc-900 px-3.5 py-2 text-xs font-medium text-white hover:bg-zinc-800 sm:self-auto"
          >
            Analyze media
          </Link>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Total Analyses" value={String(history.length)} hint="Videos processed by backend" />
          <StatCard label="Flagged Fake" value={String(fakeCount)} hint="Classified as manipulated" />
          <StatCard label="Authentic (Real)" value={String(history.length - fakeCount)} hint="Classified as authentic" />
          <StatCard
            label="Latest Confidence"
            value={latest ? `${latest.confidence}%` : "—"}
            hint={latest ? `From ${latest.filename}` : "No analyses recorded yet"}
          />
        </div>

        {latest?.frameTimeline && latest.frameTimeline.length > 0 && (
          <ConfidenceChart data={latest.frameTimeline} />
        )}

        <div className="surface-card p-6">
          <h2 className="text-sm font-semibold text-zinc-900">Recent Analyses</h2>
          {loading ? (
            <p className="mt-3 text-xs text-zinc-500">Loading analysis records…</p>
          ) : history.length === 0 ? (
            <p className="mt-3 text-xs text-zinc-500">
              No analyses recorded yet. Upload a video in the main workspace to generate a report.
            </p>
          ) : (
            <div className="mt-4 divide-y divide-stone-200">
              {history.slice(0, 6).map((item) => (
                <Link
                  key={item.id}
                  to={`/results/${item.id}`}
                  state={item}
                  className="flex items-center justify-between gap-4 py-3 text-xs transition-colors hover:bg-stone-50"
                >
                  <div>
                    <p className="font-mono font-medium text-zinc-900">{item.filename}</p>
                    <p className="mt-0.5 font-mono text-zinc-500">
                      {new Date(item.createdAt).toLocaleString()}
                    </p>
                  </div>
                  <span
                    className={`rounded px-2 py-0.5 font-mono font-semibold ${
                      item.result === "FAKE"
                        ? "bg-red-100 text-red-800"
                        : "bg-emerald-100 text-emerald-800"
                    }`}
                  >
                    {item.result} · {item.confidence}%
                  </span>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
