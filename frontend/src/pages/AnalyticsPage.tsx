import { useEffect, useState } from "react";
import { Pie, PieChart, ResponsiveContainer, Tooltip, Cell } from "recharts";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

const COLORS = ["#15803D", "#B91C1C"];

export function AnalyticsPage() {
  const [history, setHistory] = useState<Prediction[]>([]);

  useEffect(() => {
    api.get("/history").then((response) => setHistory(response.data)).catch(() => setHistory([]));
  }, []);

  const data = [
    { name: "REAL", value: history.filter((item) => item.result === "REAL").length },
    { name: "FAKE", value: history.filter((item) => item.result === "FAKE").length },
  ];

  return (
    <AppShell>
      <div className="surface-card p-6">
        <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">Distribution</p>
        <h1 className="mt-1 text-2xl font-semibold text-zinc-900">Verdict Distribution</h1>
        {history.length === 0 ? (
          <p className="mt-4 text-xs text-zinc-500">No stored predictions available to chart.</p>
        ) : (
          <div className="mt-6 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data} dataKey="value" outerRadius={100}>
                  {data.map((entry, index) => (
                    <Cell key={entry.name} fill={COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </AppShell>
  );
}
