import { useEffect, useRef, useState } from "react";
import { Upload, X, RefreshCw, AlertCircle, Film, Loader2 } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { PredictionResultView } from "../components/PredictionResultView";
import api from "../services/api";
import type { Prediction } from "../types";

const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"];
const MAX_UPLOAD_MB = 250;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
}

function getFileExtension(filename: string): string {
  const idx = filename.lastIndexOf(".");
  return idx >= 0 ? filename.slice(idx).toLowerCase() : "";
}

function validateMediaFile(candidate: File): string | null {
  const ext = getFileExtension(candidate.name);
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `Unsupported file format "${ext || "unknown"}". Supported video formats: ${ALLOWED_EXTENSIONS.join(", ")}.`;
  }
  if (candidate.size === 0) {
    return "The selected video file is empty (0 bytes). Please choose a valid video file.";
  }
  if (candidate.size > MAX_UPLOAD_BYTES) {
    return `File size (${formatFileSize(candidate.size)}) exceeds the maximum backend limit of ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

const WORKFLOW_STEPS = [
  {
    step: "01",
    title: "Media Upload & Stream Validation",
    description:
      "The FastAPI backend validates the file extension (.mp4, .mov, .avi, .mkv) and size (up to 250 MB), writes the upload to local storage, and normalizes H.264 stream encoding when ffmpeg is available.",
  },
  {
    step: "02",
    title: "Frame Sampling & MTCNN Face Alignment",
    description:
      "OpenCV decodes the video stream at a stride of every 5th frame up to a 20-frame sequence. MTCNN locates facial bounding boxes and eye landmarks, rotates frames to align inter-ocular angle, and crops faces to 224×224.",
  },
  {
    step: "03",
    title: "CNN Spatial Feature Extraction",
    description:
      "Each normalized 224×224 facial crop is passed through a TimeDistributed CNN backbone (EfficientNetB0 / forensic feature extractor) with global average pooling to encode spatial texture and blending cues.",
  },
  {
    step: "04",
    title: "Bidirectional LSTM Temporal Analysis",
    description:
      "The 20-frame sequence of spatial embeddings is analyzed by stacked Bidirectional LSTM layers (128 and 64 units) with dropout regularization to detect temporal flicker and frame-to-frame inconsistency.",
  },
  {
    step: "05",
    title: "Classification & Confidence Output",
    description:
      "A sigmoid classification head outputs the manipulation probability (threshold 0.50), mapping the video to REAL or FAKE alongside the model confidence score, class probabilities, and frame-level timeline.",
  },
];

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Manage and revoke temporary object URLs cleanly
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const handleSelectFile = (selected: File | null) => {
    // Clear any prior prediction or error so stale results are never shown for a new file
    setResult(null);
    setApiError(null);

    if (!selected) {
      setFile(null);
      setValidationError(null);
      return;
    }

    const errorMsg = validateMediaFile(selected);
    setFile(selected);
    setValidationError(errorMsg);
  };

  const handleClearFile = () => {
    setFile(null);
    setValidationError(null);
    setApiError(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    if (loading) return;
    const droppedFile = event.dataTransfer.files?.[0] ?? null;
    if (droppedFile) {
      handleSelectFile(droppedFile);
    }
  };

  const handleAnalyze = async () => {
    if (!file || loading || validationError) return;

    setLoading(true);
    setApiError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const { data } = await api.post<Prediction>("/predict", formData, {
        timeout: 180000,
      });

      if (!data || typeof data.result !== "string" || typeof data.confidence !== "number") {
        throw new Error("Received an unexpected response structure from the prediction server.");
      }

      setResult(data);
    } catch (err: any) {
      let detail = "Analysis could not be completed. Verify that the backend server is running and try again.";
      if (err?.code === "ECONNABORTED") {
        detail = "The analysis request timed out while processing the video. Try a shorter video clip.";
      } else if (err?.response?.data?.detail) {
        const rawDetail = err.response.data.detail;
        detail = Array.isArray(rawDetail)
          ? rawDetail.map((item: any) => item.msg ?? JSON.stringify(item)).join("; ")
          : String(rawDetail);
      } else if (err?.message) {
        detail = err.message;
      }
      setApiError(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppShell>
      <div className="space-y-14">
        {/* Main Analysis Workspace */}
        <section id="analyze" className="scroll-mt-20">
          <div className="max-w-2xl">
            <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
              Forensic Video Verification
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-900 sm:text-4xl">
              Verify what you see.
            </h1>
            <p className="mt-2.5 text-sm leading-relaxed text-zinc-600 sm:text-base">
              Analyze a video for signs of synthetic manipulation using your existing deepfake detection model.
            </p>
          </div>

          <div className="mt-6 surface-card p-6 sm:p-8">
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
              onChange={(e) => handleSelectFile(e.target.files?.[0] ?? null)}
              className="sr-only"
              aria-label="Upload video file"
            />

            {!file ? (
              <div
                role="button"
                tabIndex={0}
                aria-label="Select or drag and drop a video file to analyze"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(false);
                }}
                onDrop={handleDrop}
                className={`flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-6 py-12 text-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 ${
                  isDragging
                    ? "border-indigo-900 bg-indigo-950/5"
                    : "border-stone-300 bg-[#FAF9F6] hover:border-zinc-500 hover:bg-stone-100/60"
                }`}
              >
                <div className="flex h-10 w-10 items-center justify-center rounded border border-stone-200 bg-white text-zinc-700 shadow-sm">
                  <Upload className="h-5 w-5" />
                </div>
                <p className="mt-4 text-sm font-medium text-zinc-900">
                  Drag and drop a video file here, or{" "}
                  <span className="underline decoration-zinc-400 underline-offset-4">
                    browse your device
                  </span>
                </p>
                <p className="mt-1.5 font-mono text-xs text-zinc-500">
                  Supported formats: MP4, MOV, AVI, MKV · Maximum size: {MAX_UPLOAD_MB} MB
                </p>
              </div>
            ) : (
              <div className="space-y-5">
                <div className="flex flex-col justify-between gap-4 rounded-md border border-stone-200 bg-[#FAF9F6] p-4 sm:flex-row sm:items-center">
                  <div className="flex items-start gap-3 min-w-0">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded border border-stone-200 bg-white text-zinc-700">
                      <Film className="h-4 w-4" />
                    </div>
                    <div className="min-w-0">
                      <p className="truncate font-mono text-sm font-medium text-zinc-900" title={file.name}>
                        {file.name}
                      </p>
                      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-zinc-500">
                        <span>Size: {formatFileSize(file.size)}</span>
                        <span>·</span>
                        <span>
                          Type: {file.type || getFileExtension(file.name).toUpperCase().replace(".", "") || "Video"}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-1.5 rounded border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-stone-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 disabled:opacity-50"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                      Replace file
                    </button>
                    <button
                      type="button"
                      disabled={loading}
                      onClick={handleClearFile}
                      className="inline-flex items-center gap-1.5 rounded border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-red-50 hover:border-red-200 hover:text-red-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 disabled:opacity-50"
                    >
                      <X className="h-3.5 w-3.5" />
                      Remove
                    </button>
                  </div>
                </div>

                {/* Genuine Video Preview */}
                {previewUrl && !validationError && (
                  <div className="overflow-hidden rounded-md border border-stone-200 bg-zinc-950">
                    <video
                      key={previewUrl}
                      src={previewUrl}
                      controls
                      preload="metadata"
                      className="mx-auto max-h-80 w-full object-contain"
                    >
                      Your browser does not support embedded video playback.
                    </video>
                  </div>
                )}
              </div>
            )}

            {/* Validation Error */}
            {validationError && (
              <div
                role="alert"
                className="mt-4 flex items-start gap-2.5 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-800"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
                <div>
                  <p className="font-semibold">File validation error</p>
                  <p className="mt-0.5">{validationError}</p>
                </div>
              </div>
            )}

            {/* API / Processing Error */}
            {apiError && (
              <div
                role="alert"
                className="mt-4 flex items-start gap-2.5 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-800"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
                <div>
                  <p className="font-semibold">Analysis failed</p>
                  <p className="mt-0.5">{apiError}</p>
                </div>
              </div>
            )}

            {/* Honest Loading Indicator */}
            {loading && (
              <div
                role="status"
                aria-live="polite"
                className="mt-5 flex items-center gap-3 rounded-md border border-stone-200 bg-[#FAF9F6] px-4 py-3.5 text-xs text-zinc-700"
              >
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-indigo-900" />
                <div>
                  <p className="font-medium text-zinc-900">Analyzing media…</p>
                  <p className="mt-0.5 text-zinc-500">
                    Uploading file, sampling frames, detecting faces, and running CNN-LSTM sequence inference on the backend.
                  </p>
                </div>
              </div>
            )}

            {/* Submit Bar */}
            <div className="mt-6 flex flex-col gap-3 border-t border-stone-200 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs text-zinc-500">
                Files are sent directly to your configured local endpoint (<code className="font-mono text-zinc-700">POST /api/v1/predict</code>).
              </p>
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!file || Boolean(validationError) || loading}
                className="inline-flex items-center justify-center gap-2 rounded bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-zinc-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-900 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-stone-300 disabled:text-zinc-500"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing media…
                  </>
                ) : (
                  "Analyze media"
                )}
              </button>
            </div>
          </div>
        </section>

        {/* Prediction Results */}
        {result && (
          <section aria-live="polite" className="scroll-mt-20">
            <PredictionResultView prediction={result} onAnalyzeAnother={handleClearFile} />
          </section>
        )}

        {/* How It Works Section */}
        <section id="how-it-works" className="scroll-mt-20 border-t border-stone-200 pt-12">
          <div className="max-w-2xl">
            <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
              System Architecture
            </p>
            <h2 className="mt-1.5 text-xl font-semibold tracking-tight text-zinc-900 sm:text-2xl">
              How It Works
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-zinc-600">
              Each uploaded video passes through a five-stage spatial-temporal pipeline on the FastAPI backend before returning a classification verdict.
            </p>
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {WORKFLOW_STEPS.map((item) => (
              <div key={item.step} className="surface-card flex flex-col justify-between p-4">
                <div>
                  <span className="font-mono text-xs font-semibold text-indigo-900">
                    Step {item.step}
                  </span>
                  <h3 className="mt-2 text-sm font-semibold text-zinc-900">{item.title}</h3>
                  <p className="mt-2 text-xs leading-relaxed text-zinc-600">{item.description}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 rounded-md border border-stone-200 bg-[#EFECE6]/70 px-4 py-3 text-xs leading-relaxed text-zinc-600">
            <strong className="font-medium text-zinc-800">Methodological note:</strong>{" "}
            Automated deepfake detection is a probabilistic indicator rather than absolute proof of manipulation. Compression artifacts, low lighting, or extreme camera motion can influence frame-level scores.
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="scroll-mt-20 border-t border-stone-200 pt-12">
          <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
            <div>
              <p className="font-mono text-xs uppercase tracking-widest text-indigo-900">
                About DeepTrace
              </p>
              <h2 className="mt-1.5 text-xl font-semibold tracking-tight text-zinc-900 sm:text-2xl">
                Hybrid Spatial-Temporal Video Forensics
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-zinc-600">
                DeepTrace combines convolutional spatial representation learning with recurrent sequence modeling to evaluate facial integrity across time. Rather than inspecting isolated still frames in a vacuum, the pipeline evaluates whether facial geometry, boundary blending, and texture remain coherent across consecutive sampled frames.
              </p>
            </div>

            <div className="surface-card p-5">
              <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Pipeline Specifications
              </h3>
              <dl className="mt-3 divide-y divide-stone-200 text-xs">
                <div className="flex justify-between py-2">
                  <dt className="text-zinc-600">Spatial Backbone</dt>
                  <dd className="font-mono font-medium text-zinc-900">EfficientNetB0 / MobileNetV2</dd>
                </div>
                <div className="flex justify-between py-2">
                  <dt className="text-zinc-600">Temporal Sequence Model</dt>
                  <dd className="font-mono font-medium text-zinc-900">Bidirectional LSTM (128 → 64)</dd>
                </div>
                <div className="flex justify-between py-2">
                  <dt className="text-zinc-600">Face Detector & Alignment</dt>
                  <dd className="font-mono font-medium text-zinc-900">MTCNN (224×224 RGB)</dd>
                </div>
                <div className="flex justify-between py-2">
                  <dt className="text-zinc-600">Sequence Window</dt>
                  <dd className="font-mono font-medium text-zinc-900">20 frames (stride 5)</dd>
                </div>
                <div className="flex justify-between py-2">
                  <dt className="text-zinc-600">Decision Threshold</dt>
                  <dd className="font-mono font-medium text-zinc-900">0.50 (Sigmoid)</dd>
                </div>
              </dl>
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
