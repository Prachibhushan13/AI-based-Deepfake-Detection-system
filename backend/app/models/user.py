from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class UserInDB(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Literal["user", "admin"] = "user"
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

