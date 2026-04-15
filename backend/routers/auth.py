"""auth.py — Login endpoint and JWT dependency (single hardcoded admin user)."""

import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

_bearer = HTTPBearer()

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
_ALGO   = "HS256"
_DAYS   = 30

_ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", "")
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


class _Body(BaseModel):
    email: str
    password: str


def _make_token(email: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=_DAYS)
    return jwt.encode({"sub": email, "exp": exp}, _SECRET, algorithm=_ALGO)


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """FastAPI dependency — returns email if token valid, raises 401 otherwise."""
    try:
        payload = jwt.decode(creds.credentials, _SECRET, algorithms=[_ALGO])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/login")
async def login(body: _Body):
    if body.email != _ADMIN_EMAIL or body.password != _ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _make_token(body.email)
    logger.info(f"Auth: logged in — {body.email}")
    return {"token": token, "email": body.email}
