import { useEffect, useState } from "react";
import { Pie, PieChart, ResponsiveContainer, Tooltip, Cell } from "recharts";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";

const COLORS = ["#64f1d6", "#ff5e7d"];

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
      <div className="glass-panel rounded-[32px] p-8">
        <p className="text-sm uppercase tracking-[0.3em] text-cyber">Analytics</p>
        <h1 className="mt-3 font-display text-4xl font-bold">Portfolio-scale detection analytics</h1>
        <div className="mt-8 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} dataKey="value" outerRadius={110}>
                {data.map((entry, index) => (
                  <Cell key={entry.name} fill={COLORS[index]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AppShell>
  );
}

