import { useEffect, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import { ConfidenceChart } from "../components/charts/ConfidenceChart";
import api from "../services/api";
import type { Prediction } from "../types";

function getApiBaseUrl() {
  return (import.meta.env.VITE_API_URL?.replace("/api/v1", "") ?? "http://localhost:8000");
}

function buildForensicSummary(prediction: Prediction) {
  const suspiciousCount = prediction.suspiciousFrames.length;
  const peakScore = prediction.frameTimeline.length
    ? Math.max(...prediction.frameTimeline.map((point) => point.score))
    : 0;
  const averageScore = prediction.frameTimeline.length
    ? prediction.frameTimeline.reduce((sum, point) => sum + point.score, 0) / prediction.frameTimeline.length
    : 0;
  const peakFrame = prediction.frameTimeline.reduce(
    (best, current) => (current.score > best.score ? current : best),
    prediction.frameTimeline[0] ?? { frameIndex: 0, score: 0 },
  );
  const lowFrame = prediction.frameTimeline.reduce(
    (best, current) => (current.score < best.score ? current : best),
    prediction.frameTimeline[0] ?? { frameIndex: 0, score: 0 },
  );
  const highRiskFrames = prediction.frameTimeline.filter((point) => point.score >= averageScore).length;

  const verdictReason =
    prediction.result === "FAKE"
      ? "The model marked this video as likely manipulated because multiple sampled frames showed patterns that were more consistent with synthetic generation than with natural facial motion."
      : "The model marked this video as likely authentic because the sampled facial sequence stayed relatively consistent across time and did not show strong synthetic artifacts.";

  const suspiciousFramesExplanation =
    suspiciousCount > 0
      ? `The suspicious frames section highlights ${suspiciousCount} frames where the model detected the strongest anomaly signals. These are the moments most likely to contain blending artifacts, texture inconsistency, unstable facial boundaries, or temporal mismatch between adjacent frames.`
      : "No individual frames were strongly isolated as anomalous, which means the decision was driven more by the overall sequence pattern than by a single standout frame.";

  const heatmapExplanation =
    prediction.result === "FAKE"
      ? "The explainable heatmaps show which image regions most influenced the fake classification. Warmer colors suggest the model focused on areas such as cheeks, mouth edges, eyes, jaw contour, or skin transitions where manipulation artifacts often appear."
      : "The explainable heatmaps show which facial regions contributed most to the real classification. More stable attention across natural facial structure usually indicates consistent motion and texture.";

  const timelineExplanation = `The detection timeline tracks how suspiciousness changed across the analyzed sequence. The strongest signal appeared around frame ${peakFrame.frameIndex} with a score of ${peakFrame.score.toFixed(3)}, while the average sequence score was ${averageScore.toFixed(3)}.`;

  const confidenceExplanation =
    prediction.result === "FAKE"
      ? `A confidence of ${prediction.confidence}% means the model leaned toward the fake class with fake probability ${prediction.fakeProbability.toFixed(3)} versus real probability ${prediction.realProbability.toFixed(3)}.`
      : `A confidence of ${prediction.confidence}% means the model leaned toward the real class with real probability ${prediction.realProbability.toFixed(3)} versus fake probability ${prediction.fakeProbability.toFixed(3)}.`;

  return {
    verdictReason,
    suspiciousFramesExplanation,
    heatmapExplanation,
    timelineExplanation,
    confidenceExplanation,
    peakScore,
    peakFrame,
    lowFrame,
    averageScore,
    highRiskFrames,
  };
}

export function ResultsPage() {
  const { id } = useParams();
  const location = useLocation();
  const [prediction, setPrediction] = useState<Prediction | null>((location.state as Prediction) ?? null);

  useEffect(() => {
    if (prediction || !id) return;
    api.get("/history").then((response) => {
      const match = response.data.find((item: Prediction) => item.id === id);
      setPrediction(match ?? null);
    });
  }, [id, prediction]);

  if (!prediction) {
    return <AppShell><div className="glass-panel rounded-[32px] p-8">Result not found.</div></AppShell>;
  }

  const summary = buildForensicSummary(prediction);
  const apiBaseUrl = getApiBaseUrl();

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="glass-panel rounded-[32px] p-8">
          <p className="text-sm uppercase tracking-[0.3em] text-cyber">Detection Result</p>
          <h1 className="mt-3 font-display text-4xl font-bold">{prediction.result}</h1>
          <p className="mt-3 text-slate-300">Confidence score: {prediction.confidence}%</p>
          {prediction.modelMode === "mock" && (
            <p className="mt-3 rounded-2xl border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
              This result was generated in demo heuristic mode because no trained CNN-LSTM model artifact is currently loaded. It is useful for testing the pipeline, but it should not be treated as a reliable deepfake verdict.
            </p>
          )}
          {prediction.modelMode === "forensic_backbone" && (
            <p className="mt-3 rounded-2xl border border-cyber/30 bg-cyber/10 px-4 py-3 text-sm text-cyber">
              <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-cyber"></span>
              Analysis powered by MobileNetV2 Deep Forensic Backbone. Pixel-level anomaly detection and temporal feature-drift analysis are active.
            </p>
          )}
          <div className="mt-6 h-3 overflow-hidden rounded-full bg-white/10">
            <div className={`h-full ${prediction.result === "FAKE" ? "bg-alert" : "bg-cyber"}`} style={{ width: `${prediction.confidence}%` }} />
          </div>
        </div>
        <div className="grid gap-6 xl:grid-cols-2">
          <div className="glass-panel rounded-3xl p-6">
            <h3 className="font-display text-2xl font-semibold">Why This Was Flagged</h3>
            <p className="mt-4 text-slate-300">{summary.verdictReason}</p>
            <p className="mt-4 text-slate-300">{summary.confidenceExplanation}</p>
            <p className="mt-4 text-slate-300">{summary.timelineExplanation}</p>
          </div>
          <div className="glass-panel rounded-3xl p-6">
            <h3 className="font-display text-2xl font-semibold">How To Read The Evidence</h3>
            <div className="mt-4 space-y-4 text-slate-300">
              <p>
                <span className="font-medium text-white">Suspicious frames:</span>{" "}
                {summary.suspiciousFramesExplanation}
              </p>
              <p>
                <span className="font-medium text-white">Explainable heatmaps:</span>{" "}
                {summary.heatmapExplanation}
              </p>
              <p>
                <span className="font-medium text-white">Interpretation tip:</span>{" "}
                A fake result becomes more convincing when localized visual anomalies line up with repeated high scores across the timeline instead of appearing as a single random spike.
              </p>
            </div>
          </div>
        </div>
        <ConfidenceChart data={prediction.frameTimeline} />
        <div className="glass-panel rounded-3xl p-6">
          <h3 className="font-display text-2xl font-semibold">Detection Timeline Explained</h3>
          <div className="mt-4 space-y-4 text-slate-300">
            <p>
              The timeline plots the model's frame-level suspicion score across the sampled video sequence. Higher points indicate frames that contributed more strongly to a fake classification, while lower points indicate frames that looked more temporally and visually consistent.
            </p>
            <p>
              In this analysis, the strongest anomaly signal appeared at frame {summary.peakFrame.frameIndex} with a score of {summary.peakFrame.score.toFixed(3)}. The lowest signal appeared at frame {summary.lowFrame.frameIndex} with a score of {summary.lowFrame.score.toFixed(3)}.
            </p>
            <p>
              The average score across the analyzed sequence was {summary.averageScore.toFixed(3)}, and {summary.highRiskFrames} frame(s) met or exceeded that level. When several neighboring frames stay elevated instead of a single frame spiking randomly, it usually suggests a more stable manipulation pattern.
            </p>
            <p>
              Read the chart from left to right as the sampled sequence order, not absolute video time. If a cluster of higher values aligns with the suspicious frame previews and bright heatmap regions, that is the most important evidence window in the report.
            </p>
          </div>
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="glass-panel rounded-3xl p-6">
            <h3 className="font-display text-xl font-semibold">Suspicious Frames</h3>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              These frames were selected because they carried the strongest anomaly cues in the analyzed sequence. Look for warped facial edges, blurred mouth contours, inconsistent lighting, or skin texture that changes unnaturally from one frame to the next.
            </p>
            <div className="mt-4 grid grid-cols-2 gap-4">
              {prediction.suspiciousFrames.map((src) => (
                <img key={src} src={`${apiBaseUrl}${src}`} className="rounded-2xl object-cover" />
              ))}
            </div>
          </div>
          <div className="glass-panel rounded-3xl p-6">
            <h3 className="font-display text-xl font-semibold">Explainable Heatmaps</h3>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              These overlays indicate where the model concentrated its attention before producing the final verdict. Brighter regions represent stronger influence on the decision, especially around facial landmarks and synthetic blending zones.
            </p>
            <div className="mt-4 grid grid-cols-2 gap-4">
              {prediction.heatmapFrames.map((src) => (
                <img key={src} src={`${apiBaseUrl}${src}`} className="rounded-2xl object-cover" />
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
