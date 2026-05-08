from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)
client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)
db: Database = client[settings.mongodb_db]


def get_collection(name: str) -> Collection:
    return db[name]


users_collection = get_collection("users")
predictions_collection = get_collection("predictions")

try:
    client.admin.command("ping")
    users_collection.create_index("email", unique=True)
    predictions_collection.create_index("userId")
    predictions_collection.create_index("createdAt")
except Exception as exc:
    logger.warning("MongoDB unavailable during bootstrap. Continuing without index setup: %s", exc)
