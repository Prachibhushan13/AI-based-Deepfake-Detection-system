from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status

from app.middleware.auth import get_current_user, get_current_user_ws
from app.schemas.prediction import PredictionResponse
from app.services.live_detection_service import LiveDetectionSession
from app.services.prediction_service import analyze_video, delete_prediction, get_prediction_history

router = APIRouter(tags=["Predictions"])


@router.post("/upload-video", response_model=PredictionResponse)
@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), user: dict = Depends(get_current_user)) -> dict:
    return analyze_video(file, user["id"])


@router.websocket("/ws/live-detect")
async def live_detect(websocket: WebSocket) -> None:
    await websocket.accept()
    await get_current_user_ws(websocket)
    session = LiveDetectionSession()
    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") == "frame" and payload.get("data"):
                await websocket.send_json(session.push_frame(payload["data"]))
            elif payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        return


@router.get("/history", response_model=list[PredictionResponse])
async def history(user: dict = Depends(get_current_user)) -> list[dict]:
    return get_prediction_history(user["id"])


@router.delete("/history/{prediction_id}")
async def remove_history(prediction_id: str, user: dict = Depends(get_current_user)) -> dict:
    if not delete_prediction(prediction_id, user["id"]):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return {"message": "Prediction deleted"}
