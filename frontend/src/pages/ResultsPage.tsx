import { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { Loader2, AlertCircle } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { PredictionResultView } from "../components/PredictionResultView";
import api from "../services/api";
import type { Prediction } from "../types";

export function ResultsPage() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [prediction, setPrediction] = useState<Prediction | null>(
    (location.state as Prediction) ?? null,
  );
  const [loading, setLoading] = useState<boolean>(!location.state && Boolean(id));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (prediction || !id) return;
    let active = true;
    setLoading(true);
    setError(null);
    api
      .get<Prediction[]>("/history")
      .then((response) => {
        if (!active) return;
        const match = response.data.find((item) => item.id === id);
        if (match) {
          setPrediction(match);
        } else {
          setError("The requested analysis record was not found.");
        }
      })
      .catch(() => {
        if (active) {
          setError("Could not retrieve analysis record from the backend.");
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [id, prediction]);

  return (
    <AppShell>
      {loading ? (
        <div className="surface-card flex items-center gap-3 p-6 text-sm text-zinc-700">
          <Loader2 className="h-4 w-4 animate-spin text-indigo-900" />
          <span>Loading analysis report…</span>
        </div>
      ) : error || !prediction ? (
        <div className="surface-card p-6">
          <div className="flex items-start gap-2.5 text-sm text-red-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
            <div>
              <p className="font-semibold">Analysis report unavailable</p>
              <p className="mt-1 text-xs text-zinc-600">
                {error ?? "No prediction record found for this identifier."}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => navigate("/")}
            className="mt-4 rounded bg-zinc-900 px-4 py-2 text-xs font-medium text-white hover:bg-zinc-800"
          >
            Return to analysis workspace
          </button>
        </div>
      ) : (
        <PredictionResultView
          prediction={prediction}
          onAnalyzeAnother={() => navigate("/")}
        />
      )}
    </AppShell>
  );
}
