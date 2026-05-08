export type User = {
  id: string;
  name: string;
  email: string;
  role: "user" | "admin";
};

export type Prediction = {
  id: string;
  filename: string;
  modelMode: "mock" | "trained";
  result: "REAL" | "FAKE";
  confidence: number;
  realProbability: number;
  fakeProbability: number;
  suspiciousFrames: string[];
  heatmapFrames: string[];
  frameTimeline: { frameIndex: number; score: number }[];
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
      modelMode?: "mock" | "trained";
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
