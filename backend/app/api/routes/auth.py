from fastapi import APIRouter, Depends, Request

from app.middleware.rate_limit import limiter
from app.schemas.auth import AuthResponse, LoginRequest, SignupRequest
from app.services.auth_service import login_user, signup_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=AuthResponse)
@limiter.limit("10/minute")
async def signup(request: Request, payload: SignupRequest) -> dict:
    return signup_user(payload.model_dump())


@router.post("/login", response_model=AuthResponse)
@limiter.limit("10/minute")
async def login(request: Request, payload: LoginRequest) -> dict:
    return login_user(payload.email, payload.password)

