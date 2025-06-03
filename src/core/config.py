"""
Configuration management for the face recognition system.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Environment
    debug: bool = False
    log_level: str = "INFO"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Model Configuration
    model_path: Path = Path("./models")
    face_detection_confidence: float = 0.5
    face_recognition_threshold: float = 0.6
    anti_spoofing_threshold: float = 0.5
    
    # Dataset Configuration
    dataset_path: Path = Path("./datasets")
    lfw_dataset_path: Path = Path("./datasets/lfw")
    vggface2_dataset_path: Path = Path("./datasets/vggface2")
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()