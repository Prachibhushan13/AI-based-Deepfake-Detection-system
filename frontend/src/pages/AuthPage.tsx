import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { AppShell } from "../components/layout/AppShell";

type AuthPageProps = {
  mode: "login" | "register";
};

export function AuthPage({ mode }: AuthPageProps) {
  const navigate = useNavigate();
  const { login, signup, loading } = useAuth();
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      if (mode === "login") {
        await login(form.email, form.password);
      } else {
        await signup(form.name, form.email, form.password);
      }
      navigate("/");
    } catch (err: any) {
      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
          setError(detail.map((d: any) => d.msg).join(", "));
        } else {
          setError(String(detail));
        }
      } else {
        setError("An error occurred. Please try again.");
      }
    }
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-md py-8">
        <form onSubmit={onSubmit} className="surface-card p-6 sm:p-8">
          <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
            {mode === "login" ? "Sign In" : "Register"}
          </p>
          <h1 className="mt-1.5 text-xl font-semibold text-zinc-900">
            {mode === "login" ? "Access workspace session" : "Create workspace profile"}
          </h1>

          {error && (
            <div className="mt-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
              {error}
            </div>
          )}

          <div className="mt-6 space-y-4">
            {mode === "register" && (
              <div>
                <label className="block text-xs font-medium text-zinc-700">Full name</label>
                <input
                  className="mt-1 w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-indigo-900 focus:outline-none focus:ring-1 focus:ring-indigo-900"
                  placeholder="Analyst name"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-medium text-zinc-700">Email</label>
              <input
                className="mt-1 w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-indigo-900 focus:outline-none focus:ring-1 focus:ring-indigo-900"
                placeholder="name@domain.com"
                type="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-700">Password</label>
              <input
                className="mt-1 w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm text-zinc-900 focus:border-indigo-900 focus:outline-none focus:ring-1 focus:ring-indigo-900"
                placeholder="••••••••"
                type="password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-6 w-full rounded bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50"
          >
            {loading ? "Processing…" : mode === "login" ? "Sign in" : "Create profile"}
          </button>
        </form>
      </div>
    </AppShell>
  );
}
