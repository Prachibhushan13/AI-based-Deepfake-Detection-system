import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Trash2, ArrowRight, Loader2 } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

export function HistoryPage() {
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get<Prediction[]>("/history");
      setHistory(Array.isArray(response.data) ? response.data : []);
    } catch {
      setError("Unable to load analysis history from the backend.");
      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await api.delete(`/history/${id}`);
      setHistory((prev) => prev.filter((item) => item.id !== id));
    } catch {
      setError("Failed to delete record.");
    }
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
              Archive
            </p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight text-zinc-900">
              Analysis History
            </h1>
            <p className="mt-1 text-sm text-zinc-600">
              Previously processed videos stored by the backend during your workspace sessions.
            </p>
          </div>
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 self-start rounded bg-zinc-900 px-3.5 py-2 text-xs font-medium text-white transition-colors hover:bg-zinc-800 sm:self-auto"
          >
            New analysis
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-800">
            {error}
          </div>
        )}

        {loading ? (
          <div className="surface-card flex items-center gap-3 p-6 text-xs text-zinc-600">
            <Loader2 className="h-4 w-4 animate-spin text-indigo-900" />
            <span>Loading stored analysis records…</span>
          </div>
        ) : history.length === 0 ? (
          <div className="surface-card p-8 text-center">
            <p className="text-sm font-medium text-zinc-900">No stored analyses yet</p>
            <p className="mt-1 text-xs text-zinc-500">
              Analyze a video in the main workspace to view its forensic report here.
            </p>
          </div>
        ) : (
          <div className="surface-card divide-y divide-stone-200 overflow-hidden">
            {history.map((item) => {
              const isFake = item.result === "FAKE";
              return (
                <Link
                  key={item.id}
                  to={`/results/${item.id}`}
                  state={item}
                  className="flex flex-col justify-between gap-3 px-5 py-4 transition-colors hover:bg-stone-50 sm:flex-row sm:items-center"
                >
                  <div className="min-w-0">
                    <p className="truncate font-mono text-sm font-medium text-zinc-900">
                      {item.filename}
                    </p>
                    <p className="mt-0.5 font-mono text-xs text-zinc-500">
                      {new Date(item.createdAt).toLocaleString()} · Mode: {item.modelMode}
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-flex items-center rounded px-2.5 py-1 font-mono text-xs font-semibold ${
                        isFake
                          ? "border border-red-200 bg-red-50 text-red-800"
                          : "border border-emerald-200 bg-emerald-50 text-emerald-800"
                      }`}
                    >
                      {item.result} · {item.confidence}%
                    </span>
                    <button
                      type="button"
                      onClick={(e) => handleDelete(e, item.id)}
                      title="Delete record"
                      aria-label={`Delete ${item.filename}`}
                      className="rounded p-1.5 text-zinc-400 transition-colors hover:bg-red-50 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </AppShell>
  );
}
