from fastapi import HTTPException, status

from app.core.security import create_access_token, hash_password, verify_password
from app.db.mongodb import users_collection
from app.models.user import UserInDB
from app.utils.serialization import serialize_document


def signup_user(payload: dict) -> dict:
    existing = users_collection.find_one({"email": payload["email"]})
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user_data = dict(payload)
    user_data["password"] = hash_password(payload["password"])
    user = UserInDB(**user_data)
    result = users_collection.insert_one(user.model_dump())
    created = users_collection.find_one({"_id": result.inserted_id})
    serialized = serialize_document(created)
    token = create_access_token(serialized["email"])
    return {"access_token": token, "user": {k: serialized[k] for k in ["id", "name", "email", "role"]}}


def login_user(email: str, password: str) -> dict:
    user = users_collection.find_one({"email": email})
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    serialized = serialize_document(user)
    token = create_access_token(serialized["email"])
    return {"access_token": token, "user": {k: serialized[k] for k in ["id", "name", "email", "role"]}}
