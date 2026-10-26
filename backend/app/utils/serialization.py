from typing import Any


def serialize_document(document: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(document)
    serialized["id"] = str(serialized.pop("_id"))
    return serialized

