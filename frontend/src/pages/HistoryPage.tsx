import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

export function HistoryPage() {
  const [history, setHistory] = useState<Prediction[]>([]);

  useEffect(() => {
    api.get("/history").then((response) => setHistory(response.data)).catch(() => setHistory([]));
  }, []);

  return (
    <AppShell>
      <div className="glass-panel rounded-[32px] p-8">
        <p className="text-sm uppercase tracking-[0.3em] text-cyber">Detection History</p>
        <h1 className="mt-3 font-display text-4xl font-bold">Previous analysis sessions</h1>
        <div className="mt-8 space-y-4">
          {history.map((item) => (
            <Link key={item.id} to={`/results/${item.id}`} state={item} className="block rounded-2xl border border-white/10 bg-slate-950/35 p-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="font-medium">{item.filename}</p>
                  <p className="text-sm text-slate-400">{new Date(item.createdAt).toLocaleString()}</p>
                </div>
                <div className={`rounded-full px-4 py-2 text-sm ${item.result === "FAKE" ? "bg-alert/20 text-alert" : "bg-cyber/20 text-cyber"}`}>
                  {item.result} • {item.confidence}%
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </AppShell>
  );
}

