import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.core.config import settings

logger = logging.getLogger(__name__)

LOCAL_DB_FILE: Path = settings.artifact_path / "local_db.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, ObjectId):
        return {"$oid": str(obj)}
    if isinstance(obj, datetime):
        return {"$date": obj.isoformat()}
    return str(obj)


def _json_object_hook(d: dict[str, Any]) -> Any:
    if "$oid" in d and len(d) == 1:
        return ObjectId(d["$oid"])
    if "$date" in d and len(d) == 1:
        try:
            return datetime.fromisoformat(d["$date"])
        except ValueError:
            return d["$date"]
    return d


class InMemoryCursor:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def sort(self, key: str, direction: int = 1) -> "InMemoryCursor":
        reverse = direction < 0
        self._items.sort(key=lambda doc: str(doc.get(key) or ""), reverse=reverse)
        return self

    def limit(self, count: int) -> "InMemoryCursor":
        self._items = self._items[:count]
        return self

    def __iter__(self):
        return iter(self._items)


class InMemoryCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self._docs: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            if LOCAL_DB_FILE.exists():
                data = json.loads(LOCAL_DB_FILE.read_text(encoding="utf-8"), object_hook=_json_object_hook)
                return data.get(self.name, [])
        except Exception as exc:
            logger.warning("Could not load local fallback DB (%s): %s", LOCAL_DB_FILE, exc)
        return []

    def _save(self) -> None:
        try:
            LOCAL_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {}
            if LOCAL_DB_FILE.exists():
                try:
                    data = json.loads(LOCAL_DB_FILE.read_text(encoding="utf-8"), object_hook=_json_object_hook)
                except Exception:
                    data = {}
            data[self.name] = self._docs
            LOCAL_DB_FILE.write_text(json.dumps(data, default=_json_default, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not persist local fallback DB (%s): %s", LOCAL_DB_FILE, exc)

    @staticmethod
    def _matches(doc: dict[str, Any], query: dict[str, Any] | None) -> bool:
        if not query:
            return True
        return all(doc.get(k) == v for k, v in query.items())

    def create_index(self, *args: Any, **kwargs: Any) -> str:
        return "in_memory_index"

    def find_one(self, query: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self._docs = self._load()
        for doc in self._docs:
            if self._matches(doc, query):
                return copy.deepcopy(doc)
        return None

    def insert_one(self, document: dict[str, Any]) -> SimpleNamespace:
        self._docs = self._load()
        record = copy.deepcopy(document)
        record.setdefault("_id", ObjectId())
        self._docs.append(record)
        self._save()
        return SimpleNamespace(inserted_id=record["_id"])

    def find(
        self,
        query: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
    ) -> InMemoryCursor:
        self._docs = self._load()
        matched = [copy.deepcopy(doc) for doc in self._docs if self._matches(doc, query)]
        if projection:
            include_keys = {k for k, v in projection.items() if v}
            if include_keys:
                include_keys.add("_id")
                matched = [{k: v for k, v in doc.items() if k in include_keys} for doc in matched]
        return InMemoryCursor(matched)

    def delete_one(self, query: dict[str, Any]) -> SimpleNamespace:
        self._docs = self._load()
        for idx, doc in enumerate(self._docs):
            if self._matches(doc, query):
                del self._docs[idx]
                self._save()
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    def count_documents(self, query: dict[str, Any] | None = None) -> int:
        self._docs = self._load()
        return sum(1 for doc in self._docs if self._matches(doc, query))

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._docs = self._load()
        for stage in pipeline:
            if "$group" in stage:
                group_spec = stage["$group"]
                group_key = group_spec.get("_id", "").lstrip("$")
                buckets: dict[Any, list[dict[str, Any]]] = {}
                for doc in self._docs:
                    buckets.setdefault(doc.get(group_key), []).append(doc)
                results: list[dict[str, Any]] = []
                for key_val, docs in buckets.items():
                    row: dict[str, Any] = {"_id": key_val}
                    for field_name, expr in group_spec.items():
                        if field_name == "_id":
                            continue
                        if isinstance(expr, dict) and "$avg" in expr:
                            target = str(expr["$avg"]).lstrip("$")
                            vals = [float(d[target]) for d in docs if d.get(target) is not None]
                            row[field_name] = sum(vals) / len(vals) if vals else 0.0
                    results.append(row)
                return results
        return []


client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000)
db: Database = client[settings.mongodb_db]


def get_collection(name: str) -> Collection | InMemoryCollection:
    return db[name]


users_collection: Collection | InMemoryCollection = get_collection("users")
predictions_collection: Collection | InMemoryCollection = get_collection("predictions")

try:
    client.admin.command("ping")
    users_collection.create_index("email", unique=True)
    predictions_collection.create_index("userId")
    predictions_collection.create_index("createdAt")
except Exception as exc:
    logger.warning("MongoDB unavailable (%s). Using persistent local fallback collections.", exc)
    users_collection = InMemoryCollection("users")
    predictions_collection = InMemoryCollection("predictions")
