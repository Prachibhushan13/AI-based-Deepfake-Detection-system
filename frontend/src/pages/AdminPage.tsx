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
      <div className="glass-panel rounded-[32px] p-8">
        <p className="text-sm uppercase tracking-[0.3em] text-cyber">Admin Panel</p>
        <h1 className="mt-3 font-display text-4xl font-bold">System-wide operational intelligence</h1>
        {stats ? (
          <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl bg-slate-950/40 p-5">Users: {stats.totalUsers}</div>
            <div className="rounded-2xl bg-slate-950/40 p-5">Predictions: {stats.totalPredictions}</div>
            <div className="rounded-2xl bg-slate-950/40 p-5">Fake: {stats.fakeCount}</div>
            <div className="rounded-2xl bg-slate-950/40 p-5">Avg confidence: {stats.averageConfidence}%</div>
          </div>
        ) : (
          <p className="mt-6 text-slate-300">Admin statistics require an administrator account.</p>
        )}
      </div>
    </AppShell>
  );
}

