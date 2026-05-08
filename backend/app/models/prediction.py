from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PredictionInDB(BaseModel):
    userId: str
    filename: str
    modelMode: str = "trained"
    result: str
    confidence: float
    realProbability: float
    fakeProbability: float
    suspiciousFrames: list[str] = Field(default_factory=list)
    frameTimeline: list[dict] = Field(default_factory=list)
    heatmapFrames: list[str] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
