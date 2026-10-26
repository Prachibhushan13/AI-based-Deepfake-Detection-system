import { useState } from "react";
import { RotateCcw, FileText, CheckCircle2, ShieldAlert } from "lucide-react";
import { ConfidenceChart } from "./charts/ConfidenceChart";
import api, { getServerRootUrl } from "../services/api";
import type { Prediction, SuspiciousFrameDetail } from "../types";

type PredictionResultViewProps = {
  prediction: Prediction;
  onAnalyzeAnother: () => void;
};

function formatModelMode(mode: string): string {
  switch (mode) {
    case "trained_hybrid":
      return "Hybrid CNN-LSTM (Trained)";
    case "trained_xception":
      return "Xception Spatial Baseline (Trained)";
    case "transformers_vit":
      return "Vision Transformer (ViT)";
    case "forensic_backbone":
      return "CNN-LSTM Forensic Backbone";
    default:
      return mode;
  }
}

export function PredictionResultView({ prediction, onAnalyzeAnother }: PredictionResultViewProps) {
  const [downloadingReport, setDownloadingReport] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  const isFake = prediction.result === "FAKE";
  const serverRoot = getServerRootUrl();

  const hasProbabilities =
    typeof prediction.realProbability === "number" &&
    typeof prediction.fakeProbability === "number";

  const hasTimeline = Array.isArray(prediction.frameTimeline) && prediction.frameTimeline.length > 0;
  const videoMeta = prediction.videoMetadata;

  // Build unified suspicious frame cards from suspiciousFrameDetails or fallback to suspiciousFrames
  const suspiciousDetails: SuspiciousFrameDetail[] =
    Array.isArray(prediction.suspiciousFrameDetails) && prediction.suspiciousFrameDetails.length > 0
      ? prediction.suspiciousFrameDetails
      : (prediction.suspiciousFrames ?? []).map((frameUrl, idx) => {
          const timelineEntry = prediction.frameTimeline?.[idx];
          return {
            slot: idx,
            sampleIndex: idx,
            frameIndex: timelineEntry?.frameIndex ?? idx,
            timestamp: timelineEntry?.timestamp ?? 0,
            timestampFormatted: timelineEntry?.timestampFormatted ?? "00:00.00",
            score: timelineEntry?.score ?? prediction.fakeProbability,
            frameUrl,
            heatmapUrl: prediction.heatmapFrames?.[idx],
          };
        });

  const handleDownloadReport = async () => {
    if (!prediction.id) return;
    setDownloadingReport(true);
    setReportError(null);
    try {
      const { data } = await api.get(`/report/${prediction.id}`);
      if (data?.reportUrl) {
        window.open(`${serverRoot}${data.reportUrl}`, "_blank", "noopener,noreferrer");
      }
    } catch {
      setReportError("Unable to generate PDF report for this analysis.");
    } finally {
      setDownloadingReport(false);
    }
  };

  return (
    <div className="space-y-6" role="region" aria-label="Analysis Result">
      {/* Primary Verdict Card */}
      <div className="surface-card overflow-hidden">
        <div
          className={`border-b px-6 py-4 flex flex-wrap items-center justify-between gap-4 ${
            isFake
              ? "border-red-200 bg-red-50/70"
              : "border-emerald-200 bg-emerald-50/70"
          }`}
        >
          <div className="flex items-center gap-2.5">
            {isFake ? (
              <ShieldAlert className="h-5 w-5 text-red-700" aria-hidden="true" />
            ) : (
              <CheckCircle2 className="h-5 w-5 text-emerald-700" aria-hidden="true" />
            )}
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-700">
              Analysis Complete
            </span>
          </div>

          <div className="flex items-center gap-2">
            {prediction.id && (
              <button
                type="button"
                onClick={handleDownloadReport}
                disabled={downloadingReport}
                className="inline-flex items-center gap-1.5 rounded border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-stone-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 disabled:opacity-50"
              >
                <FileText className="h-3.5 w-3.5" />
                {downloadingReport ? "Preparing PDF…" : "Export PDF"}
              </button>
            )}
            <button
              type="button"
              onClick={onAnalyzeAnother}
              className="inline-flex items-center gap-1.5 rounded bg-zinc-900 px-3.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-zinc-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Analyze another file
            </button>
          </div>
        </div>

        <div className="p-6">
          {reportError && (
            <div className="mb-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {reportError}
            </div>
          )}

          <div className="grid gap-6 md:grid-cols-3">
            {/* Classification Verdict */}
            <div className="rounded-md border border-stone-200 bg-[#FAF9F6] p-4">
              <p className="text-xs font-mono uppercase tracking-wider text-zinc-500">
                Prediction
              </p>
              <div className="mt-2 flex items-baseline gap-2.5">
                <span
                  className={`font-mono text-2xl font-bold tracking-tight ${
                    isFake ? "text-red-700" : "text-emerald-700"
                  }`}
                >
                  {prediction.result}
                </span>
                <span
                  className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                    isFake
                      ? "bg-red-100 text-red-800"
                      : "bg-emerald-100 text-emerald-800"
                  }`}
                >
                  {isFake ? "Manipulation indicators detected" : "Consistent with authentic video"}
                </span>
              </div>
            </div>

            {/* Confidence Score */}
            <div className="rounded-md border border-stone-200 bg-[#FAF9F6] p-4">
              <div className="flex items-baseline justify-between">
                <p className="text-xs font-mono uppercase tracking-wider text-zinc-500">
                  Confidence Score
                </p>
                <span className="font-mono text-xl font-bold text-zinc-900">
                  {prediction.confidence}%
                </span>
              </div>
              <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-stone-200">
                <div
                  className={`h-full transition-all ${
                    isFake ? "bg-red-600" : "bg-emerald-600"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, prediction.confidence))}%` }}
                />
              </div>
              <p className="mt-2 text-[11px] text-zinc-500">
                Aggregated from frame-level sequence predictions (threshold: 0.50).
              </p>
            </div>

            {/* Class Probabilities */}
            {hasProbabilities && (
              <div className="rounded-md border border-stone-200 bg-[#FAF9F6] p-4">
                <p className="text-xs font-mono uppercase tracking-wider text-zinc-500">
                  Class Probabilities
                </p>
                <div className="mt-2.5 space-y-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-600">Real (Authentic)</span>
                    <span className="font-mono font-medium text-zinc-900">
                      {(prediction.realProbability * 100).toFixed(1)}% ({prediction.realProbability.toFixed(4)})
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-600">Fake (Synthetic)</span>
                    <span className="font-mono font-medium text-zinc-900">
                      {(prediction.fakeProbability * 100).toFixed(1)}% ({prediction.fakeProbability.toFixed(4)})
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Genuine Analysis & Video Stream Metadata */}
          <div className="mt-6 border-t border-stone-200 pt-4">
            <dl className="grid grid-cols-2 gap-4 text-xs sm:grid-cols-3 lg:grid-cols-6">
              <div>
                <dt className="font-mono uppercase text-zinc-500">Source File</dt>
                <dd className="mt-1 truncate font-mono font-medium text-zinc-900" title={prediction.filename}>
                  {prediction.filename}
                </dd>
              </div>
              <div>
                <dt className="font-mono uppercase text-zinc-500">Pipeline Mode</dt>
                <dd className="mt-1 font-medium text-zinc-900">
                  {formatModelMode(prediction.modelMode)}
                </dd>
              </div>
              <div>
                <dt className="font-mono uppercase text-zinc-500">Sampled Frames</dt>
                <dd className="mt-1 font-mono font-medium text-zinc-900">
                  {videoMeta ? `${videoMeta.sampledFrames} / ${videoMeta.totalFrames}` : `${prediction.frameTimeline.length} frames`}
                </dd>
              </div>
              {videoMeta && (
                <>
                  <div>
                    <dt className="font-mono uppercase text-zinc-500">Frame Rate / Stride</dt>
                    <dd className="mt-1 font-mono font-medium text-zinc-900">
                      {videoMeta.fps} FPS (stride {videoMeta.stride})
                    </dd>
                  </div>
                  <div>
                    <dt className="font-mono uppercase text-zinc-500">Duration / Resolution</dt>
                    <dd className="mt-1 font-mono font-medium text-zinc-900">
                      {videoMeta.durationSeconds}s · {videoMeta.resolution}
                    </dd>
                  </div>
                </>
              )}
              {prediction.createdAt && (
                <div>
                  <dt className="font-mono uppercase text-zinc-500">Analyzed At</dt>
                  <dd className="mt-1 font-mono text-zinc-700">
                    {new Date(prediction.createdAt).toLocaleTimeString()}
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      </div>

      {/* Frame-level predictions chart & table */}
      {hasTimeline && <ConfidenceChart data={prediction.frameTimeline} />}

      {/* Dynamic Suspicious Frames Section */}
      <div className="surface-card p-6">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold text-zinc-900">
              Suspicious Frames ({suspiciousDetails.length})
            </h3>
            <p className="mt-0.5 text-xs text-zinc-500">
              Frames from the uploaded video whose individual manipulation score met or exceeded the 0.50 suspicion threshold.
            </p>
          </div>
          <span className="font-mono text-xs text-zinc-500">
            Threshold: score ≥ 0.5000
          </span>
        </div>

        {suspiciousDetails.length === 0 ? (
          <div className="mt-4 rounded-md border border-stone-200 bg-[#FAF9F6] p-6 text-center">
            <p className="text-xs font-medium text-zinc-800">
              No suspicious frames exceeded the anomaly threshold (0.50) in this video.
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              All {prediction.frameTimeline.length} sampled frames scored below 0.50 during spatial-temporal evaluation.
            </p>
          </div>
        ) : (
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {suspiciousDetails.map((item) => (
              <div
                key={`${item.frameIndex}-${item.slot}`}
                className="overflow-hidden rounded-md border border-stone-200 bg-[#FAF9F6]"
              >
                <div className="grid grid-cols-2 gap-px bg-stone-200">
                  <div className="relative bg-zinc-950">
                    <img
                      src={`${serverRoot}${item.frameUrl}`}
                      alt={`Source frame #${item.frameIndex} at ${item.timestampFormatted}`}
                      className="aspect-video w-full object-cover"
                      loading="lazy"
                    />
                    <span className="absolute bottom-1.5 left-1.5 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[10px] text-white">
                      Original
                    </span>
                  </div>
                  {item.heatmapUrl ? (
                    <div className="relative bg-zinc-950">
                      <img
                        src={`${serverRoot}${item.heatmapUrl}`}
                        alt={`Anomaly map for frame #${item.frameIndex}`}
                        className="aspect-video w-full object-cover"
                        loading="lazy"
                      />
                      <span className="absolute bottom-1.5 left-1.5 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[10px] text-white">
                        Spectral Map
                      </span>
                    </div>
                  ) : null}
                </div>

                <div className="flex items-center justify-between px-3.5 py-2.5 font-mono text-xs">
                  <div>
                    <span className="font-semibold text-zinc-900">
                      Frame #{item.frameIndex}
                    </span>
                    <span className="ml-2 text-zinc-500">
                      @ {item.timestampFormatted} ({item.timestamp.toFixed(2)}s)
                    </span>
                  </div>
                  <span className="rounded bg-red-100 px-2 py-0.5 font-semibold text-red-800">
                    Score: {item.score.toFixed(4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
