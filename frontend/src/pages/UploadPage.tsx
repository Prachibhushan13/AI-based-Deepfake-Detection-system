import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { LiveDetectionMessage } from "../types";

type Mode = "upload" | "live";

function getSocketUrl() {
  const apiBase = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
  const root = apiBase.replace("/api/v1", "");
  return `${root.replace("http://", "ws://").replace("https://", "wss://")}/api/v1/ws/live-detect`;
}

function buildLiveExplanation(result: Extract<LiveDetectionMessage, { type: "prediction" }>) {
  const realPercent = result.realProbability * 100;
  const fakePercent = result.fakeProbability * 100;
  const averageTimelineScore = result.timeline.length
    ? result.timeline.reduce((sum, point) => sum + point.score, 0) / result.timeline.length
    : 0;
  const elevatedFrames = result.timeline.filter((point) => point.score >= averageTimelineScore).length;
  const stabilityScore = Math.max(0, Math.min(100, realPercent - result.suspiciousFrameCount * 4));

  const verdictLine =
    result.result === "REAL"
      ? "The current sliding frame window looks more consistent with a natural face sequence than with a manipulated one."
      : "The current sliding frame window shows stronger synthetic or inconsistent facial cues than a natural sequence.";

  const evidenceLine =
    result.result === "REAL"
      ? `Real probability is ${realPercent.toFixed(1)}%, suspicious frames are limited to ${result.suspiciousFrameCount}, and the timeline does not show a strong cluster of repeated anomaly spikes.`
      : `Fake probability is ${fakePercent.toFixed(1)}%, suspicious frames in the current window are ${result.suspiciousFrameCount}, and elevated timeline values suggest repeated anomaly cues instead of a single random spike.`;

  const scaleLabel =
    result.result === "REAL"
      ? "Authenticity Scale"
      : "Manipulation Risk Scale";

  return {
    realPercent,
    fakePercent,
    averageTimelineScore,
    elevatedFrames,
    stabilityScore,
    verdictLine,
    evidenceLine,
    scaleLabel,
  };
}

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<Mode>("upload");
  const [liveEnabled, setLiveEnabled] = useState(false);
  const [liveStatus, setLiveStatus] = useState("Idle");
  const [liveResult, setLiveResult] = useState<LiveDetectionMessage | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const navigate = useNavigate();

  const overlayColor = useMemo(() => {
    if (liveResult?.type !== "prediction") return "border-cyber/40 bg-slate-950/55";
    return liveResult.result === "FAKE" ? "border-alert/60 bg-slate-950/65" : "border-cyber/60 bg-slate-950/65";
  }, [liveResult]);
  const liveExplanation = useMemo(
    () => (liveResult?.type === "prediction" ? buildLiveExplanation(liveResult) : null),
    [liveResult],
  );

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await api.post("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      navigate(`/results/${data.id}`, { state: data });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!liveEnabled) return;

    let intervalId: number | null = null;

    const startLiveDetection = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 540 } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        const token = localStorage.getItem("deepfake_token");
        const ws = new WebSocket(`${getSocketUrl()}?token=${encodeURIComponent(token ?? "")}`);
        socketRef.current = ws;

        ws.onopen = () => {
          setLiveStatus("Streaming webcam frames for live analysis.");
          intervalId = window.setInterval(() => {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            if (!video || !canvas || ws.readyState !== WebSocket.OPEN || video.readyState < 2) return;
            const context = canvas.getContext("2d");
            if (!context) return;
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 360;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frame = canvas.toDataURL("image/jpeg", 0.7);
            ws.send(JSON.stringify({ type: "frame", data: frame }));
          }, 200);
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data) as LiveDetectionMessage;
          setLiveResult(data);
          if (data.type !== "pong" && "message" in data) {
            setLiveStatus(data.message);
          }
        };

        ws.onerror = () => {
          setLiveStatus("Live detection socket error. Please retry.");
        };

        ws.onclose = () => {
          setLiveStatus("Live stream stopped.");
        };
      } catch (error) {
        console.error(error);
        setLiveStatus("Unable to access webcam. Check browser permissions.");
      }
    };

    startLiveDetection();

    return () => {
      if (intervalId) window.clearInterval(intervalId);
      socketRef.current?.close();
      socketRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setLiveResult(null);
    };
  }, [liveEnabled]);

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="glass-panel rounded-[32px] p-8">
          <p className="text-sm uppercase tracking-[0.3em] text-cyber">Upload & Analyze</p>
          <h1 className="mt-3 font-display text-4xl font-bold">Inspect a video or webcam feed for deepfake traces</h1>
          <div className="mt-8 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => {
                setMode("upload");
                setLiveEnabled(false);
              }}
              className={`rounded-full px-5 py-2 text-sm transition ${mode === "upload" ? "bg-cyber text-slate-950" : "border border-white/10 text-white"}`}
            >
              Upload Video
            </button>
            <button
              type="button"
              onClick={() => setMode("live")}
              className={`rounded-full px-5 py-2 text-sm transition ${mode === "live" ? "bg-cyber text-slate-950" : "border border-white/10 text-white"}`}
            >
              Live Webcam
            </button>
          </div>
        </div>

        {mode === "upload" ? (
          <div className="glass-panel rounded-[32px] p-8">
            <div className="rounded-[28px] border border-dashed border-cyber/40 bg-slate-950/40 p-10 text-center">
              <input type="file" accept=".mp4,.mov,.avi,.mkv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <p className="mt-4 text-slate-300">Drag and drop or browse to upload a video for forensic analysis.</p>
              <button type="button" onClick={submit} disabled={!file || loading} className="mt-6 rounded-full bg-cyber px-6 py-3 font-semibold text-slate-950">
                {loading ? "Analyzing..." : "Run Deepfake Detection"}
              </button>
            </div>
          </div>
        ) : (
          <div className="grid gap-6 xl:grid-cols-[1.4fr_0.8fr]">
            <div className="glass-panel rounded-[32px] p-6">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="font-display text-2xl font-semibold">Live Webcam Detection</h2>
                  <p className="mt-2 text-sm text-slate-300">Uses `getUserMedia()`, samples frames at ~5 fps, and streams them over WebSocket to the FastAPI backend.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setLiveEnabled((current) => !current)}
                  className={`rounded-full px-5 py-2 text-sm font-semibold transition ${liveEnabled ? "bg-alert text-white" : "bg-cyber text-slate-950"}`}
                >
                  {liveEnabled ? "Stop Live Detection" : "Start Live Detection"}
                </button>
              </div>

              <div className="relative mt-6 overflow-hidden rounded-[28px] border border-white/10 bg-slate-950/60">
                <video ref={videoRef} playsInline muted className="aspect-video w-full object-cover" />
                <canvas ref={canvasRef} className="hidden" />
                <div className={`absolute left-4 top-4 max-w-sm rounded-2xl border px-4 py-3 backdrop-blur ${overlayColor}`}>
                  <p className="text-xs uppercase tracking-[0.25em] text-cyber">Live Confidence Overlay</p>
                  {liveResult?.type === "prediction" ? (
                    <div className="mt-3">
                      <p className={`font-display text-2xl font-bold ${liveResult.result === "FAKE" ? "text-alert" : "text-cyber"}`}>
                        {liveResult.result}
                      </p>
                      <p className="mt-1 text-sm text-slate-200">Confidence: {liveResult.confidence.toFixed(2)}%</p>
                      <p className="mt-1 text-sm text-slate-300">Real probability: {(liveResult.realProbability * 100).toFixed(2)}%</p>
                      <p className="mt-1 text-sm text-slate-300">Fake probability: {(liveResult.fakeProbability * 100).toFixed(2)}%</p>
                      <p className="mt-1 text-sm text-slate-300">Window size: {liveResult.frameCount} frames</p>
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-200">{liveStatus}</p>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="glass-panel rounded-[32px] p-6">
                <h3 className="font-display text-xl font-semibold">Live Stream Status</h3>
                <p className="mt-4 text-slate-300">{liveStatus}</p>
                {liveResult?.type === "prediction" && liveExplanation && (
                  <div className="mt-6 space-y-4 text-sm text-slate-300">
                    <div>
                      <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-400">
                        <span>{liveExplanation.scaleLabel}</span>
                        <span>{liveResult.result === "REAL" ? `${liveExplanation.realPercent.toFixed(1)}% real` : `${liveExplanation.fakePercent.toFixed(1)}% fake`}</span>
                      </div>
                      <div className="mt-3 h-3 overflow-hidden rounded-full bg-white/10">
                        <div
                          className={`h-full ${liveResult.result === "REAL" ? "bg-cyber" : "bg-alert"}`}
                          style={{ width: `${liveResult.result === "REAL" ? liveExplanation.realPercent : liveExplanation.fakePercent}%` }}
                        />
                      </div>
                    </div>

                    <p>{liveExplanation.verdictLine}</p>
                    <p>{liveExplanation.evidenceLine}</p>

                    <div className="rounded-2xl border border-white/10 bg-slate-950/35 p-4">
                      <p><span className="font-medium text-white">Stability score:</span> {liveExplanation.stabilityScore.toFixed(1)}/100</p>
                      <p className="mt-2"><span className="font-medium text-white">Suspicious frames:</span> {liveResult.suspiciousFrameCount}</p>
                      <p className="mt-2"><span className="font-medium text-white">Elevated timeline frames:</span> {liveExplanation.elevatedFrames}</p>
                      <p className="mt-2"><span className="font-medium text-white">Inference mode:</span> sliding window of recent frames streamed from the browser</p>
                    </div>

                    <p className="text-xs leading-6 text-slate-400">
                      A `REAL` indication is stronger when the real probability stays high, suspicious frames remain low, and the timeline avoids repeated high anomaly clusters across neighboring frames.
                    </p>
                  </div>
                )}
              </div>
              <div className="glass-panel rounded-[32px] p-6">
                <h3 className="font-display text-xl font-semibold">How It Works</h3>
                <div className="mt-4 space-y-3 text-sm leading-6 text-slate-300">
                  <p>1. The browser captures a webcam `MediaStream` using WebRTC `getUserMedia()`.</p>
                  <p>2. A hidden canvas samples the feed at roughly 5 fps and converts frames to base64 JPEG chunks.</p>
                  <p>3. The frontend streams those chunks over WebSocket to FastAPI.</p>
                  <p>4. The backend maintains a sliding frame window and runs inference repeatedly to update the overlay.</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}
