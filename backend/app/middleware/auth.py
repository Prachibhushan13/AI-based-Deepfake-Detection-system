from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token, hash_password
from app.db.mongodb import users_collection
from app.utils.serialization import serialize_document

security = HTTPBearer(auto_error=False)
DEFAULT_WORKSPACE_EMAIL = "analyst@deeptrace.ai"


def _get_or_create_user_by_email(email: str) -> dict:
    user = users_collection.find_one({"email": email})
    if not user:
        from app.models.user import UserInDB
        fallback_user = UserInDB(
            name=email.split("@")[0],
            email=email,
            password=hash_password("DeepTraceWorkspace2026!"),
            role="user",
        )
        inserted = users_collection.insert_one(fallback_user.model_dump())
        user = users_collection.find_one({"_id": inserted.inserted_id})
    return serialize_document(user)


def get_user_from_token(token: str | None) -> dict:
    if not token:
        return _get_or_create_user_by_email(DEFAULT_WORKSPACE_EMAIL)
    try:
        payload = decode_access_token(token)
        email = payload.get("sub") or DEFAULT_WORKSPACE_EMAIL
    except ValueError:
        email = DEFAULT_WORKSPACE_EMAIL
    return _get_or_create_user_by_email(email)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    token = credentials.credentials if credentials else None
    return get_user_from_token(token)


async def get_current_user_ws(websocket: WebSocket) -> dict:
    token = websocket.query_params.get("token")
    return get_user_from_token(token)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user
