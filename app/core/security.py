import base64
import hashlib

import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from jose import jwt, JWTError
from app.core.config import settings

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Generates bcrypt hash for password."""
    # Truncate to 72 bytes if needed (bcrypt limitation)
    pwd_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def create_access_token(subject: Any, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a signed JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "iat": datetime.now(timezone.utc)
    }
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decodes JWT access token and returns payload if valid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def _fernet() -> Fernet:
    """Fernet instance keyed from JWT_SECRET (stable across restarts)."""
    key = hashlib.sha256(settings.JWT_SECRET.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret (e.g. a provider API key) for storage at rest."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a stored secret; raises ValueError if the token is invalid."""
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Stored secret could not be decrypted (JWT_SECRET changed?).") from exc
