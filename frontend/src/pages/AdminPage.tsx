import { useEffect, useState } from "react";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { AdminStats } from "../types";

export function AdminPage() {
  const [stats, setStats] = useState<AdminStats | null>(null);

  useEffect(() => {
    api.get("/admin/stats").then((response) => setStats(response.data)).catch(() => setStats(null));
  }, []);

  return (
    <AppShell>
      <div className="surface-card p-6">
        <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">Administration</p>
        <h1 className="mt-1 text-2xl font-semibold text-zinc-900">System Statistics</h1>
        {stats ? (
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 text-xs">
            <div className="rounded border border-stone-200 bg-[#FAF9F6] p-4">
              <span className="font-mono uppercase text-zinc-500">Users</span>
              <p className="mt-1 font-mono text-lg font-semibold text-zinc-900">{stats.totalUsers}</p>
            </div>
            <div className="rounded border border-stone-200 bg-[#FAF9F6] p-4">
              <span className="font-mono uppercase text-zinc-500">Predictions</span>
              <p className="mt-1 font-mono text-lg font-semibold text-zinc-900">{stats.totalPredictions}</p>
            </div>
            <div className="rounded border border-stone-200 bg-[#FAF9F6] p-4">
              <span className="font-mono uppercase text-zinc-500">Flagged Fake</span>
              <p className="mt-1 font-mono text-lg font-semibold text-zinc-900">{stats.fakeCount}</p>
            </div>
            <div className="rounded border border-stone-200 bg-[#FAF9F6] p-4">
              <span className="font-mono uppercase text-zinc-500">Average Confidence</span>
              <p className="mt-1 font-mono text-lg font-semibold text-zinc-900">{stats.averageConfidence}%</p>
            </div>
          </div>
        ) : (
          <p className="mt-4 text-xs text-zinc-600">Administrator statistics require an admin role.</p>
        )}
      </div>
    </AppShell>
  );
}
