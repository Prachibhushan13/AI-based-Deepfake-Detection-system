import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

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
      navigate("/dashboard");
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
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={onSubmit} className="glass-panel w-full max-w-md rounded-[32px] p-8">
        <p className="text-sm uppercase tracking-[0.3em] text-cyber">{mode === "login" ? "Welcome Back" : "Create Account"}</p>
        <h1 className="mt-4 font-display text-4xl font-bold">
          {mode === "login" ? "Access the detection lab" : "Build your forensic workspace"}
        </h1>
        {error && (
          <div className="mt-4 rounded-xl bg-red-500/10 p-4 border border-red-500/20 text-red-400">
            {error}
          </div>
        )}
        <div className="mt-8 space-y-4">
          {mode === "register" && (
            <input
              className="w-full rounded-2xl border border-white/10 bg-slate-950/50 px-4 py-3"
              placeholder="Full name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          )}
          <input
            className="w-full rounded-2xl border border-white/10 bg-slate-950/50 px-4 py-3"
            placeholder="Email"
            type="email"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
          />
          <input
            className="w-full rounded-2xl border border-white/10 bg-slate-950/50 px-4 py-3"
            placeholder="Password"
            type="password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
          />
        </div>
        <button type="submit" disabled={loading} className="mt-6 w-full rounded-2xl bg-cyber px-5 py-3 font-semibold text-slate-950">
          {loading ? "Please wait..." : mode === "login" ? "Login" : "Register"}
        </button>
      </form>
    </div>
  );
}

