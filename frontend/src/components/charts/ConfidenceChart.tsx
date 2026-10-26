import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TimelinePoint } from "../../types";

type ConfidenceChartProps = {
  data: TimelinePoint[];
};

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const point: TimelinePoint = payload[0].payload;
  const hasTimestamp = typeof point.timestamp === "number";

  return (
    <div className="rounded-md border border-stone-300 bg-white p-3 font-mono text-xs text-zinc-900 shadow-sm">
      <p className="font-semibold text-zinc-900">
        Frame #{point.frameIndex}
        {hasTimestamp ? ` · ${point.timestampFormatted ?? `${point.timestamp?.toFixed(2)}s`}` : ""}
      </p>
      <div className="mt-1.5 space-y-1 text-[11px]">
        <div className="flex justify-between gap-4">
          <span className="text-zinc-500">Combined score:</span>
          <span className="font-semibold text-indigo-950">{Number(point.score).toFixed(4)}</span>
        </div>
        {typeof point.spatialScore === "number" && (
          <div className="flex justify-between gap-4">
            <span className="text-zinc-500">Spatial (CNN):</span>
            <span>{point.spatialScore.toFixed(4)}</span>
          </div>
        )}
        {typeof point.temporalScore === "number" && (
          <div className="flex justify-between gap-4">
            <span className="text-zinc-500">Temporal (LSTM):</span>
            <span>{point.temporalScore.toFixed(4)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function ConfidenceChart({ data }: ConfidenceChartProps) {
  if (!data || data.length === 0) {
    return null;
  }

  const hasTimestamps = data.some((d) => typeof d.timestamp === "number");

  return (
    <div className="surface-card p-6 space-y-6">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-zinc-900">
            Dynamic Frame-by-Frame Detection Timeline
          </h3>
          <p className="text-xs text-zinc-500">
            Per-frame manipulation score across the video timeline (threshold 0.50 marked by dashed line).
          </p>
        </div>
        <span className="font-mono text-xs text-zinc-500">
          {data.length} sampled frame{data.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="forensicScoreFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#312E81" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#312E81" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#E4E2DD" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey={hasTimestamps ? "timestamp" : "frameIndex"}
              stroke="#71717A"
              tick={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
              tickFormatter={(val) =>
                hasTimestamps ? `${Number(val).toFixed(2)}s` : `#${val}`
              }
            />
            <YAxis
              domain={[0, 1]}
              stroke="#71717A"
              tick={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
              tickFormatter={(val) => Number(val).toFixed(1)}
            />
            <ReferenceLine y={0.5} stroke="#B91C1C" strokeDasharray="4 4" strokeOpacity={0.6} />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#312E81"
              strokeWidth={1.75}
              fill="url(#forensicScoreFill)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Frame-level breakdown table */}
      <details className="group border-t border-stone-200 pt-4">
        <summary className="cursor-pointer select-none font-mono text-xs font-medium text-zinc-700 hover:text-zinc-900">
          View frame-by-frame score log ({data.length} frames)
        </summary>
        <div className="mt-3 max-h-60 overflow-y-auto rounded border border-stone-200">
          <table className="w-full text-left font-mono text-xs">
            <thead className="sticky top-0 border-b border-stone-200 bg-[#FAF9F6] text-[11px] uppercase text-zinc-500">
              <tr>
                <th className="px-3 py-2">Sample</th>
                <th className="px-3 py-2">Source Frame</th>
                <th className="px-3 py-2">Timestamp</th>
                <th className="px-3 py-2">Spatial</th>
                <th className="px-3 py-2">Temporal</th>
                <th className="px-3 py-2">Combined Score</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-200 bg-white">
              {data.map((point, idx) => {
                const isSuspicious = point.score >= 0.5;
                return (
                  <tr key={`${point.frameIndex}-${idx}`} className="hover:bg-stone-50">
                    <td className="px-3 py-1.5 text-zinc-500">{point.sampleIndex ?? idx}</td>
                    <td className="px-3 py-1.5 font-medium text-zinc-900">#{point.frameIndex}</td>
                    <td className="px-3 py-1.5 text-zinc-600">
                      {point.timestampFormatted ??
                        (typeof point.timestamp === "number" ? `${point.timestamp.toFixed(2)}s` : "—")}
                    </td>
                    <td className="px-3 py-1.5 text-zinc-600">
                      {typeof point.spatialScore === "number" ? point.spatialScore.toFixed(4) : "—"}
                    </td>
                    <td className="px-3 py-1.5 text-zinc-600">
                      {typeof point.temporalScore === "number" ? point.temporalScore.toFixed(4) : "—"}
                    </td>
                    <td className="px-3 py-1.5 font-semibold text-zinc-900">
                      {point.score.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5">
                      <span
                        className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                          isSuspicious
                            ? "bg-red-100 text-red-800"
                            : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {isSuspicious ? "FLAGGED" : "NORMAL"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
