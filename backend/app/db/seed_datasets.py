import uuid
from datetime import datetime, timedelta, timezone

from app.db.mongodb import predictions_collection, users_collection
from app.models.prediction import PredictionInDB


def seed_dataset_results():
    # 1. Get or create a sample user to own these predictions
    user = users_collection.find_one({"email": "prachi@example.com"})
    if not user:
        print("Sample user not found. Please sign up first or create a user manually.")
        return
    
    user_id = str(user["_id"])
    
    # 2. Define sample dataset entries
    datasets = [
        {
            "name": "FaceForensics++",
            "samples": [
                {"file": "001_003.mp4", "result": "FAKE", "conf": 98.42, "fake": 0.9842, "real": 0.0158},
                {"file": "742.mp4", "result": "REAL", "conf": 96.10, "fake": 0.0390, "real": 0.9610},
                {"file": "891_912.mp4", "result": "FAKE", "conf": 92.15, "fake": 0.9215, "real": 0.0785},
            ]
        },
        {
            "name": "Celeb-DF",
            "samples": [
                {"file": "id0_id1_0002.mp4", "result": "FAKE", "conf": 99.12, "fake": 0.9912, "real": 0.0088},
                {"file": "id16_0000.mp4", "result": "REAL", "conf": 97.45, "fake": 0.0255, "real": 0.9745},
                {"file": "id21_id23_0009.mp4", "result": "FAKE", "conf": 88.30, "fake": 0.8830, "real": 0.1170},
            ]
        },
        {
            "name": "DFDC (Preview)",
            "samples": [
                {"file": "aagfhgj.mp4", "result": "FAKE", "conf": 94.20, "fake": 0.9420, "real": 0.0580},
                {"file": "bggte.mp4", "result": "REAL", "conf": 91.80, "fake": 0.0820, "real": 0.9180},
            ]
        }
    ]
    
    print(f"Seeding {sum(len(d['samples']) for d in datasets)} dataset records...")
    
    now = datetime.now(timezone.utc)
    
    for ds in datasets:
        for i, sample in enumerate(ds["samples"]):
            # Create a realistic timeline
            timeline = []
            for f in range(20):
                base = sample["fake"] if sample["result"] == "FAKE" else sample["real"]
                score = max(0.01, min(0.99, base + (0.05 * (f % 3 - 1))))
                timeline.append({"frameIndex": f, "score": round(score, 4)})
            
            doc = PredictionInDB(
                userId=user_id,
                filename=f"[{ds['name']}] {sample['file']}",
                modelMode="trained",
                result=sample["result"],
                confidence=sample["conf"],
                realProbability=sample["real"],
                fakeProbability=sample["fake"],
                suspiciousFrames=[],
                frameTimeline=timeline,
                heatmapFrames=[],
                createdAt=now - timedelta(days=2, hours=i)
            )
            predictions_collection.insert_one(doc.model_dump())
            
    print("Seeding complete.")


if __name__ == "__main__":
    seed_dataset_results()
