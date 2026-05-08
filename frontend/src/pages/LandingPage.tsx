import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Brain, Radar, ShieldCheck } from "lucide-react";

const features = [
  { icon: Brain, title: "Hybrid CNN-LSTM", text: "Spatial face forensics with temporal inconsistency modeling." },
  { icon: Radar, title: "Explainable AI", text: "Heatmaps, suspicious frames, and confidence timelines for every report." },
  { icon: ShieldCheck, title: "Production Security", text: "JWT auth, rate limiting, validated uploads, and container deployment." },
];

export function LandingPage() {
  return (
    <div className="grid-overlay min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-8 lg:px-8">
        <header className="flex items-center justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-cyber">AI Forensics Platform</p>
            <h1 className="font-display text-2xl font-bold">DeepSight AI</h1>
          </div>
          <div className="flex gap-3">
            <Link to="/login" className="rounded-full border border-white/10 px-5 py-2 text-sm">Login</Link>
            <Link to="/register" className="rounded-full bg-cyber px-5 py-2 text-sm font-semibold text-slate-950">Get Started</Link>
          </div>
        </header>

        <section className="flex flex-1 flex-col justify-center py-16">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="max-w-4xl">
            <p className="text-sm uppercase tracking-[0.35em] text-cyber/80">Final Year Project • Research Prototype • Portfolio Ready</p>
            <h2 className="mt-6 font-display text-5xl font-bold leading-tight md:text-7xl">
              Detect manipulated videos with forensic depth and production-grade UX.
            </h2>
            <p className="mt-6 max-w-2xl text-lg text-slate-300">
              Upload videos, extract faces, analyze frame sequences with a Hybrid CNN-LSTM, and review explainable results in a modern analytics dashboard.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <Link to="/register" className="rounded-full bg-cyber px-6 py-3 font-semibold text-slate-950">
                Launch Console
              </Link>
              <Link to="/dashboard" className="rounded-full border border-white/10 px-6 py-3">
                View Demo Dashboard
              </Link>
            </div>
          </motion.div>

          <div className="mt-16 grid gap-6 md:grid-cols-3">
            {features.map(({ icon: Icon, title, text }) => (
              <div key={title} className="glass-panel rounded-[28px] p-6">
                <Icon className="text-cyber" />
                <h3 className="mt-5 font-display text-xl font-semibold">{title}</h3>
                <p className="mt-3 text-slate-300">{text}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

