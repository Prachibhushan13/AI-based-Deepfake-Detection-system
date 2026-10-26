import { Link, NavLink } from "react-router-dom";
import { Activity, FileClock, LayoutDashboard, LogOut, ShieldAlert, UploadCloud, SplitSquareHorizontal } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/upload", label: "Analyze", icon: UploadCloud },
  { to: "/compare", label: "Model Compare", icon: SplitSquareHorizontal },
  { to: "/analytics", label: "Analytics", icon: Activity },
  { to: "/history", label: "History", icon: FileClock },
  { to: "/admin", label: "Admin", icon: ShieldAlert },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const { logout, user } = useAuth();

  return (
    <div className="min-h-screen">
      <div className="mx-auto flex max-w-7xl gap-6 px-4 py-6 lg:px-8">
        <aside className="glass-panel hidden w-72 rounded-[28px] p-6 lg:block">
          <Link to="/" className="font-display text-2xl font-bold">
            DeepSight AI
          </Link>
          <p className="mt-3 text-sm text-slate-300">
            Hybrid CNN-LSTM for forensic video integrity analysis.
          </p>
          <nav className="mt-8 space-y-2">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-2xl px-4 py-3 text-sm transition ${
                    isActive ? "bg-cyber/15 text-cyber" : "text-slate-300 hover:bg-white/5"
                  }`
                }
              >
                <Icon size={18} />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="mt-10 rounded-2xl border border-white/10 bg-slate-950/40 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-cyber/70">Signed in</p>
            <p className="mt-2 font-medium">{user?.name}</p>
            <p className="text-sm text-slate-400">{user?.email}</p>
            <button
              type="button"
              onClick={logout}
              className="mt-4 inline-flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm text-white transition hover:border-cyber hover:text-cyber"
            >
              <LogOut size={16} />
              Logout
            </button>
          </div>
        </aside>
        <main className="flex-1">{children}</main>
      </div>
    </div>
  );
}

