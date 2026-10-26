import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
const SERVER_ROOT_URL = API_BASE_URL.replace(/\/api\/v1\/?$/, "");

const WORKSPACE_CREDENTIALS = {
  name: "DeepTrace Analyst",
  email: "analyst@deeptrace.ai",
  password: "DeepTraceWorkspace2026!",
};

let tokenPromise: Promise<string | null> | null = null;

export function getServerRootUrl(): string {
  return SERVER_ROOT_URL;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${SERVER_ROOT_URL}/health`, { timeout: 4000 });
    return response.data?.status === "ok";
  } catch {
    return false;
  }
}

export async function ensureSessionToken(forceRefresh = false): Promise<string | null> {
  if (!forceRefresh) {
    const existing = localStorage.getItem("deepfake_token");
    if (existing) {
      return existing;
    }
  }

  if (tokenPromise) {
    return tokenPromise;
  }

  tokenPromise = (async () => {
    try {
      const loginResp = await axios.post(`${API_BASE_URL}/auth/login`, {
        email: WORKSPACE_CREDENTIALS.email,
        password: WORKSPACE_CREDENTIALS.password,
      });
      const token = loginResp.data?.access_token;
      if (token) {
        localStorage.setItem("deepfake_token", token);
        localStorage.setItem("deepfake_user", JSON.stringify(loginResp.data.user));
        return token;
      }
    } catch {
      try {
        const signupResp = await axios.post(`${API_BASE_URL}/auth/signup`, WORKSPACE_CREDENTIALS);
        const token = signupResp.data?.access_token;
        if (token) {
          localStorage.setItem("deepfake_token", token);
          localStorage.setItem("deepfake_user", JSON.stringify(signupResp.data.user));
          return token;
        }
      } catch {
        return null;
      }
    }
    return null;
  })().finally(() => {
    tokenPromise = null;
  });

  return tokenPromise;
}

const api = axios.create({
  baseURL: API_BASE_URL,
});

api.interceptors.request.use(async (config) => {
  let token = localStorage.getItem("deepfake_token");
  const isAuthRoute = config.url?.includes("/auth/login") || config.url?.includes("/auth/signup");
  if (!token && !isAuthRoute) {
    token = await ensureSessionToken();
  }
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    const isAuthRoute =
      originalRequest?.url?.includes("/auth/login") || originalRequest?.url?.includes("/auth/signup");
    if (error.response?.status === 401 && originalRequest && !originalRequest._retry && !isAuthRoute) {
      originalRequest._retry = true;
      const newToken = await ensureSessionToken(true);
      if (newToken) {
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      }
    }
    return Promise.reject(error);
  },
);

export default api;
