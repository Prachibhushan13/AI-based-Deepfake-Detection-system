import { useState } from "react";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function ComparePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const { data } = await api.post("/predict", formData);
      setResult(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Comparison analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const renderComparisonCard = (title: string, data: any, strokeColor: string) => {
    if (!data) return null;
    const isFake = data.prediction === "FAKE";
    return (
      <div className="surface-card p-5 flex flex-col gap-3">
        <div>
          <h3 className="text-sm font-semibold text-zinc-900">{title}</h3>
          <div className="mt-2 flex items-baseline justify-between">
            <span className={`font-mono text-lg font-bold ${isFake ? "text-red-700" : "text-emerald-700"}`}>
              {data.prediction}
            </span>
            <span className="font-mono text-xs text-zinc-600">{data.confidence.toFixed(2)}% confidence</span>
          </div>
        </div>
        <div className="h-36 w-full mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.timeline}>
              <XAxis dataKey="frameIndex" hide />
              <YAxis domain={[0, 1]} hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #D6D3CD",
                  borderRadius: "6px",
                  fontSize: "12px",
                  fontFamily: "JetBrains Mono, monospace",
                }}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke={strokeColor}
                fill={strokeColor}
                fillOpacity={0.12}
                strokeWidth={1.75}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="surface-card p-6">
          <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
            Architecture Comparison
          </p>
          <h1 className="mt-1 text-2xl font-semibold text-zinc-900">
            Spatial vs. Temporal Model Comparison
          </h1>
          <p className="mt-1.5 text-sm text-zinc-600">
            Compare spatial-only (CNN), temporal-only (LSTM), and unified hybrid (CNN-LSTM) outputs returned by the backend.
          </p>

          <form onSubmit={onSubmit} className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">
            <input
              type="file"
              accept=".mp4,.mov,.avi,.mkv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full max-w-sm rounded border border-stone-300 bg-[#FAF9F6] px-3 py-2 text-xs text-zinc-700"
            />
            <button
              type="submit"
              disabled={!file || loading}
              className="rounded bg-zinc-900 px-4 py-2 text-xs font-medium text-white hover:bg-zinc-800 disabled:opacity-50"
            >
              {loading ? "Analyzing…" : "Run comparison"}
            </button>
          </form>

          {error && (
            <div className="mt-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
              {error}
            </div>
          )}
        </div>

        {result && result.modelsComparison && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {renderComparisonCard("CNN (Spatial Only)", result.modelsComparison.CNN, "#15803D")}
            {renderComparisonCard("LSTM (Temporal Only)", result.modelsComparison.LSTM, "#B45309")}
            {renderComparisonCard("CNN-LSTM (Hybrid)", result.modelsComparison.CNN_LSTM, "#312E81")}
          </div>
        )}
      </div>
    </AppShell>
  );
}
