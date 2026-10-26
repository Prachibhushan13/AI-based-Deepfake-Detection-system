export type User = {
  id: string;
  name: string;
  email: string;
  role: "user" | "admin";
};

export type Prediction = {
  id: string;
  filename: string;
  modelMode: "mock" | "trained" | "trained_torch_temporal" | "forensic_backbone";
  result: "REAL" | "FAKE";
  confidence: number;
  realProbability: number;
  fakeProbability: number;
  suspiciousFrames: string[];
  heatmapFrames: string[];
  frameTimeline: { frameIndex: number; score: number }[];
  modelsComparison?: {
    CNN: { confidence: number; prediction: string; timeline: { frameIndex: number; score: number }[] };
    LSTM: { confidence: number; prediction: string; timeline: { frameIndex: number; score: number }[] };
    CNN_LSTM: { confidence: number; prediction: string; timeline: { frameIndex: number; score: number }[] };
  };
  facesAnalysis?: { faceId: number; fakeProbability: number; boundingBox?: number[] }[];
  audioAnalysis?: { audioFakeProbability: number; lipSyncDiscrepancy: number };
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
      modelMode?: "mock" | "trained" | "trained_torch_temporal" | "forensic_backbone";
      confidence: number;
      fakeProbability: number;
      realProbability: number;
      timeline: { frameIndex: number; score: number }[];
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
