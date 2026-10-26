export type User = {
  id: string;
  name: string;
  email: string;
  role: "user" | "admin";
};

export type TimelinePoint = {
  sampleIndex?: number;
  frameIndex: number;
  timestamp?: number;
  timestampFormatted?: string;
  score: number;
  spatialScore?: number;
  temporalScore?: number;
};

export type SuspiciousFrameDetail = {
  slot: number;
  sampleIndex: number;
  frameIndex: number;
  timestamp: number;
  timestampFormatted: string;
  score: number;
  frameUrl: string;
  heatmapUrl?: string;
};

export type VideoMetadata = {
  fps: number;
  totalFrames: number;
  sampledFrames: number;
  stride: number;
  durationSeconds: number;
  resolution: string;
};

export type Prediction = {
  id: string;
  filename: string;
  modelMode: "trained" | "trained_hybrid" | "trained_xception" | "transformers_vit" | "forensic_backbone" | string;
  result: "REAL" | "FAKE";
  confidence: number;
  realProbability: number;
  fakeProbability: number;
  suspiciousFrames: string[];
  suspiciousFrameDetails?: SuspiciousFrameDetail[];
  heatmapFrames: string[];
  frameTimeline: TimelinePoint[];
  videoMetadata?: VideoMetadata | null;
  modelsComparison?: {
    CNN: { confidence: number; prediction: string; timeline: TimelinePoint[] };
    LSTM: { confidence: number; prediction: string; timeline: TimelinePoint[] };
    CNN_LSTM: { confidence: number; prediction: string; timeline: TimelinePoint[] };
  };
  facesAnalysis?: { faceId: number; fakeProbability: number; boundingBox?: number[] }[];
  audioAnalysis?: { audioFakeProbability: number; lipSyncDiscrepancy: number } | null;
  createdAt: string;
};

export type LiveDetectionMessage =
  | {
      type: "status";
      ready: boolean;
      frameCount: number;
      requiredFrames: number;
      message: string;
    }
  | {
      type: "prediction";
      ready: boolean;
      frameCount: number;
      requiredFrames: number;
      result: "REAL" | "FAKE";
      modelMode?: string;
      confidence: number;
      fakeProbability: number;
      realProbability: number;
      timeline: TimelinePoint[];
      suspiciousFrameCount: number;
      message: string;
    }
  | {
      type: "pong";
    };

export type AdminStats = {
  totalUsers: number;
  totalPredictions: number;
  fakeCount: number;
  realCount: number;
  averageConfidence: number;
  recentTrend: { date: string; result: string }[];
};
