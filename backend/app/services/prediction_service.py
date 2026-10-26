import logging
import uuid

from bson import ObjectId
from bson.errors import InvalidId

from app.db.mongodb import predictions_collection
from app.ml.gradcam import generate_heatmap_assets
from app.ml.inference import predict_video
from app.models.prediction import PredictionInDB
from app.services.video_service import compress_video, extract_frames_with_metadata, save_upload
from app.utils.serialization import serialize_document

logger = logging.getLogger(__name__)


def normalize_prediction(document: dict) -> dict:
    normalized = serialize_document(document)
    normalized.setdefault("modelMode", "forensic_backbone")
    normalized.setdefault("suspiciousFrames", [])
    normalized.setdefault("suspiciousFrameDetails", [])
    normalized.setdefault("heatmapFrames", [])
    normalized.setdefault("frameTimeline", [])
    normalized.setdefault("videoMetadata", None)
    return normalized


def analyze_video(file, user_id: str) -> dict:
    logger.info("Starting video analysis for filename='%s', user_id='%s'", file.filename, user_id)
    saved_path = save_upload(file)
    compressed_path = compress_video(saved_path)
    frames, video_metadata = extract_frames_with_metadata(compressed_path)
    inference = predict_video(frames, filename=file.filename, video_metadata=video_metadata)

    asset_prefix = str(uuid.uuid4())
    suspicious_paths, heatmap_paths, suspicious_details = generate_heatmap_assets(
        frames,
        inference["suspiciousIndices"],
        asset_prefix,
        timeline=inference["timeline"],
    )

    clean_video_meta = {
        k: v
        for k, v in video_metadata.items()
        if k not in ("frameIndices", "timestamps")
    }

    logger.info(
        "Completed inference for '%s': result=%s, confidence=%.2f%%, modelMode=%s, sampledFrames=%d, suspiciousFrames=%d",
        file.filename,
        inference["result"],
        inference["confidence"],
        inference["modelMode"],
        len(frames),
        len(suspicious_details),
    )

    document = PredictionInDB(
        userId=user_id,
        filename=file.filename,
        modelMode=inference["modelMode"],
        result=inference["result"],
        confidence=inference["confidence"],
        realProbability=inference["realProbability"],
        fakeProbability=inference["fakeProbability"],
        suspiciousFrames=suspicious_paths,
        suspiciousFrameDetails=suspicious_details,
        heatmapFrames=heatmap_paths,
        frameTimeline=inference["timeline"],
        videoMetadata=clean_video_meta,
        modelsComparison=inference.get("modelsComparison"),
        facesAnalysis=inference.get("facesAnalysis"),
        audioAnalysis=inference.get("audioAnalysis"),
    )
    inserted = predictions_collection.insert_one(document.model_dump())
    created = predictions_collection.find_one({"_id": inserted.inserted_id})
    return normalize_prediction(created)


def get_prediction_history(user_id: str) -> list[dict]:
    cursor = predictions_collection.find({"userId": user_id}).sort("createdAt", -1)
    return [normalize_prediction(item) for item in cursor]


def get_prediction_by_id(prediction_id: str, user_id: str | None = None) -> dict | None:
    try:
        query = {"_id": ObjectId(prediction_id)}
    except InvalidId:
        return None
    if user_id:
        query["userId"] = user_id
    item = predictions_collection.find_one(query)
    return normalize_prediction(item) if item else None


def delete_prediction(prediction_id: str, user_id: str) -> bool:
    try:
        object_id = ObjectId(prediction_id)
    except InvalidId:
        return False
    deleted = predictions_collection.delete_one({"_id": object_id, "userId": user_id})
    return deleted.deleted_count > 0
