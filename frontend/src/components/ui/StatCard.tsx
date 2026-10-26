type StatCardProps = {
  label: string;
  value: string;
  hint: string;
};

export function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <div className="surface-card p-5">
      <p className="text-xs font-mono uppercase tracking-wider text-zinc-500">{label}</p>
      <p className="mt-2 font-mono text-2xl font-semibold text-zinc-900">{value}</p>
      <p className="mt-1 text-xs text-zinc-600">{hint}</p>
    </div>
  );
}
