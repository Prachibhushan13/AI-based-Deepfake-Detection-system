from fastapi import APIRouter, Depends, HTTPException, status

from app.middleware.auth import get_current_user
from app.services.prediction_service import get_prediction_by_id
from app.services.report_service import build_pdf_report

router = APIRouter(tags=["Reports"])


@router.get("/report/{prediction_id}")
async def get_report(prediction_id: str, user: dict = Depends(get_current_user)) -> dict:
    prediction = get_prediction_by_id(prediction_id, user["id"])
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    report_path = build_pdf_report(prediction)
    return {"reportUrl": report_path, "prediction": prediction}

