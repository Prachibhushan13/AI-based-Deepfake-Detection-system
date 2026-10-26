from fastapi import APIRouter, Depends

from app.db.mongodb import predictions_collection, users_collection
from app.middleware.auth import require_admin
from app.schemas.prediction import AdminStatsResponse

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/stats", response_model=AdminStatsResponse)
async def admin_stats(user: dict = Depends(require_admin)) -> dict:
    total_users = users_collection.count_documents({})
    total_predictions = predictions_collection.count_documents({})
    fake_count = predictions_collection.count_documents({"result": "FAKE"})
    real_count = predictions_collection.count_documents({"result": "REAL"})
    pipeline = [
        {"$group": {"_id": "$result", "avgConfidence": {"$avg": "$confidence"}}},
    ]
    grouped = list(predictions_collection.aggregate(pipeline))
    average_confidence = sum(item["avgConfidence"] for item in grouped) / len(grouped) if grouped else 0
    recent = list(predictions_collection.find({}, {"createdAt": 1, "result": 1}).sort("createdAt", -1).limit(10))
    trend = [{"date": str(item["createdAt"]), "result": item["result"]} for item in recent]
    return {
        "totalUsers": total_users,
        "totalPredictions": total_predictions,
        "fakeCount": fake_count,
        "realCount": real_count,
        "averageConfidence": round(average_confidence, 2),
        "recentTrend": trend,
    }

