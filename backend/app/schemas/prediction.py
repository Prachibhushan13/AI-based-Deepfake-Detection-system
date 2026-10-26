from datetime import datetime

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    id: str
    filename: str
    modelMode: str
    result: str
    confidence: float
    realProbability: float
    fakeProbability: float
    suspiciousFrames: list[str]
    suspiciousFrameDetails: list[dict] = Field(default_factory=list)
    heatmapFrames: list[str]
    frameTimeline: list[dict]
    videoMetadata: dict | None = None
    modelsComparison: dict | None = None
    facesAnalysis: list[dict] | None = None
    audioAnalysis: dict | None = None
    createdAt: datetime


class AdminStatsResponse(BaseModel):
    totalUsers: int
    totalPredictions: int
    fakeCount: int
    realCount: int
    averageConfidence: float
    recentTrend: list[dict]
