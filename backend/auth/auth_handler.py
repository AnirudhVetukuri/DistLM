from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import os
from ..config import JWT_CONFIG

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"  # "user", "node_owner", "admin"

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        JWT_CONFIG["secret_key"], 
        algorithm=JWT_CONFIG["algorithm"]
    )
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, 
            JWT_CONFIG["secret_key"], 
            algorithms=[JWT_CONFIG["algorithm"]]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"))
    except JWTError:
        raise credentials_exception
        
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

# Temporary in-memory user storage (replace with database)
users_db = {}

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None

def create_user(username: str, password: str, role: str = "user") -> User:
    """Create new user"""
    if username in users_db:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(password)
    user_dict = {
        "username": username,
        "hashed_password": hashed_password,
        "role": role,
        "disabled": False
    }
    users_db[username] = user_dict
    return User(**user_dict)

def check_permissions(user: User, required_role: str) -> bool:
    """Check if user has required role"""
    role_hierarchy = {
        "admin": 3,
        "node_owner": 2,
        "user": 1
    }
    
    user_level = role_hierarchy.get(user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    return user_level >= required_level 