import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type ConfidenceChartProps = {
  data: { frameIndex: number; score: number }[];
};

export function ConfidenceChart({ data }: ConfidenceChartProps) {
  return (
    <div className="glass-panel rounded-3xl p-6">
      <h3 className="font-display text-xl font-semibold">Detection Timeline</h3>
      <div className="mt-6 h-72">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="cyberLine" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#64f1d6" stopOpacity={0.9} />
                <stop offset="100%" stopColor="#64f1d6" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="frameIndex" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Area type="monotone" dataKey="score" stroke="#64f1d6" fill="url(#cyberLine)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

