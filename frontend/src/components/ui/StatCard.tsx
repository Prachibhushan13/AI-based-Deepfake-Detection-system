import { motion } from "framer-motion";

type StatCardProps = {
  label: string;
  value: string;
  hint: string;
};

export function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-3xl p-6"
    >
      <p className="text-sm uppercase tracking-[0.25em] text-cyber/70">{label}</p>
      <p className="mt-4 font-display text-3xl font-semibold">{value}</p>
      <p className="mt-2 text-sm text-slate-300">{hint}</p>
    </motion.div>
  );
}

