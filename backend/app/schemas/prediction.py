from datetime import datetime

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: str
    filename: str
    modelMode: str
    result: str
    confidence: float
    realProbability: float
    fakeProbability: float
    suspiciousFrames: list[str]
    heatmapFrames: list[str]
    frameTimeline: list[dict]
    modelsComparison: dict | None = None
    createdAt: datetime


class AdminStatsResponse(BaseModel):
    totalUsers: int
    totalPredictions: int
    fakeCount: int
    realCount: int
    averageConfidence: float
    recentTrend: list[dict]
