import { createContext, useContext, useEffect, useState } from "react";
import api, { ensureSessionToken } from "../services/api";
import type { User } from "../types";

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const DEFAULT_WORKSPACE_USER: User = {
  id: "workspace",
  name: "DeepTrace Analyst",
  email: "analyst@deeptrace.ai",
  role: "user",
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    const stored = localStorage.getItem("deepfake_user");
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch {
        return DEFAULT_WORKSPACE_USER;
      }
    }
    return DEFAULT_WORKSPACE_USER;
  });
  const [loading, setLoading] = useState(false);

  const applySession = (token: string, nextUser: User) => {
    localStorage.setItem("deepfake_token", token);
    localStorage.setItem("deepfake_user", JSON.stringify(nextUser));
    setUser(nextUser);
  };

  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      const { data } = await api.post("/auth/login", { email, password });
      applySession(data.access_token, data.user);
    } finally {
      setLoading(false);
    }
  };

  const signup = async (name: string, email: string, password: string) => {
    setLoading(true);
    try {
      const { data } = await api.post("/auth/signup", { name, email, password });
      applySession(data.access_token, data.user);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("deepfake_token");
    localStorage.removeItem("deepfake_user");
    setUser(DEFAULT_WORKSPACE_USER);
  };

  useEffect(() => {
    ensureSessionToken().then(() => {
      const stored = localStorage.getItem("deepfake_user");
      if (stored) {
        try {
          setUser(JSON.parse(stored));
        } catch {
          // keep default workspace user
        }
      }
    });
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
