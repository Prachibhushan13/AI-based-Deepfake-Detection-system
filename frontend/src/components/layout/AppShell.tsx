import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { checkBackendHealth } from "../../services/api";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    let active = true;
    const verify = async () => {
      const ok = await checkBackendHealth();
      if (active) {
        setBackendOnline(ok);
      }
    };
    verify();
    const interval = window.setInterval(verify, 15000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const scrollToSection = (sectionId: string) => {
    const isWorkspacePage = location.pathname === "/" || location.pathname === "/upload";
    if (!isWorkspacePage) {
      navigate(`/#${sectionId}`);
      setTimeout(() => {
        document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth" });
      }, 100);
      return;
    }
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    } else if (sectionId === "analyze") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F6F2] text-zinc-900">
      <header className="sticky top-0 z-30 border-b border-stone-200/90 bg-[#F7F6F2]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6 lg:px-8">
          <div className="flex items-center gap-6">
            <Link
              to="/"
              className="group flex items-center gap-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 rounded"
            >
              <span className="flex h-7 w-7 items-center justify-center rounded border border-zinc-800 bg-zinc-900 text-xs font-mono font-semibold tracking-tight text-white">
                DT
              </span>
              <div className="flex items-baseline gap-2">
                <span className="text-base font-semibold tracking-tight text-zinc-900">
                  DeepTrace
                </span>
                <span className="hidden text-xs font-mono uppercase tracking-wider text-zinc-500 sm:inline">
                  Forensic Media Lab
                </span>
              </div>
            </Link>

            <nav aria-label="Primary" className="flex items-center gap-1 sm:gap-2">
              <button
                type="button"
                onClick={() => scrollToSection("analyze")}
                className={`rounded px-2.5 py-1.5 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 ${
                  location.pathname === "/" || location.pathname === "/upload"
                    ? "text-zinc-900 hover:bg-stone-200/60"
                    : "text-zinc-600 hover:bg-stone-200/60 hover:text-zinc-900"
                }`}
              >
                Analyze
              </button>
              <button
                type="button"
                onClick={() => scrollToSection("how-it-works")}
                className="rounded px-2.5 py-1.5 text-sm font-medium text-zinc-600 transition-colors hover:bg-stone-200/60 hover:text-zinc-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900"
              >
                How It Works
              </button>
              <button
                type="button"
                onClick={() => scrollToSection("about")}
                className="rounded px-2.5 py-1.5 text-sm font-medium text-zinc-600 transition-colors hover:bg-stone-200/60 hover:text-zinc-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900"
              >
                About
              </button>
              <Link
                to="/history"
                className={`rounded px-2.5 py-1.5 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 ${
                  location.pathname === "/history"
                    ? "bg-stone-200/80 text-zinc-900"
                    : "text-zinc-600 hover:bg-stone-200/60 hover:text-zinc-900"
                }`}
              >
                History
              </Link>
            </nav>
          </div>

          {backendOnline !== null && (
            <div
              className="flex items-center gap-2 rounded border border-stone-200 bg-white px-2.5 py-1 text-xs font-mono text-zinc-600"
              title={
                backendOnline
                  ? "Verified via GET /health endpoint"
                  : "Backend API /health endpoint did not respond"
              }
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendOnline ? "bg-emerald-600" : "bg-amber-600"
                }`}
              />
              <span className="hidden sm:inline">
                {backendOnline ? "Inference engine ready" : "Backend offline"}
              </span>
              <span className="sm:hidden">{backendOnline ? "Online" : "Offline"}</span>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>

      <footer className="mt-16 border-t border-stone-200/90 bg-[#EFECE6]/60">
        <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="max-w-md">
              <p className="text-sm font-semibold text-zinc-900">DeepTrace</p>
              <p className="mt-1 text-xs leading-relaxed text-zinc-600">
                Digital media authenticity and deepfake analysis tool powered by a hybrid CNN-LSTM spatial-temporal inference pipeline.
              </p>
            </div>
            <p className="max-w-lg text-xs leading-relaxed text-zinc-500">
              <strong className="font-medium text-zinc-700">Responsible use notice:</strong>{" "}
              Automated detection results provide a probabilistic technical indicator, not definitive proof of manipulation or authenticity. Always evaluate findings alongside source provenance and domain review.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
