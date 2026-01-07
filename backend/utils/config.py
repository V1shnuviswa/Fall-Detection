"""
Configuration management for the fall detection system.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Configuration
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    workers: int = Field(default=4, alias="WORKERS")
    
    # Model Configuration
    model_path: Path = Field(default=Path("./trained_models/lstm_fall_detector.pth"), alias="MODEL_PATH")
    yolo_model: str = Field(default="yolov8m-pose.pt", alias="YOLO_MODEL")
    device: Literal["cuda", "cpu"] = Field(default="cuda", alias="DEVICE")
    
    # Detection Thresholds
    fall_threshold: float = Field(default=0.88, alias="FALL_THRESHOLD")
    posture_threshold: float = Field(default=0.20, alias="POSTURE_THRESHOLD")
    motion_threshold: float = Field(default=0.15, alias="MOTION_THRESHOLD")
    temporal_weight: float = Field(default=0.65, alias="TEMPORAL_WEIGHT")
    posture_weight: float = Field(default=0.20, alias="POSTURE_WEIGHT")
    motion_weight: float = Field(default=0.15, alias="MOTION_WEIGHT")
    
    # Temporal Settings
    buffer_size: int = Field(default=30, alias="BUFFER_SIZE")  # frames in temporal window
    stride: int = Field(default=5, alias="STRIDE")
    target_fps: int = Field(default=10, alias="TARGET_FPS")
    
    # Physics Validation
    min_lstm_probability: float = Field(default=0.85, alias="MIN_LSTM_PROBABILITY")
    max_torso_angle: float = Field(default=30.0, alias="MAX_TORSO_ANGLE")  # degrees
    max_bbox_ratio: float = Field(default=1.0, alias="MAX_BBOX_RATIO")
    inactivity_threshold: int = Field(default=5, alias="INACTIVITY_THRESHOLD")  # seconds
    
    # Alert Settings
    confirmation_seconds: int = Field(default=2, alias="CONFIRMATION_SECONDS")
    alert_timeout: int = Field(default=15, alias="ALERT_TIMEOUT")
    max_alerts_per_day: int = Field(default=10, alias="MAX_ALERTS_PER_DAY")
    
    # Camera Settings
    frame_width: int = Field(default=640, alias="FRAME_WIDTH")
    frame_height: int = Field(default=480, alias="FRAME_HEIGHT")
    jpeg_quality: int = Field(default=85, alias="JPEG_QUALITY")
    
    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        alias="CORS_ORIGINS"
    )
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Path = Field(default=Path("./logs/fall_detection.log"), alias="LOG_FILE")
    
    # Privacy
    save_poses: bool = Field(default=False, alias="SAVE_POSES")
    save_images: bool = Field(default=False, alias="SAVE_IMAGES")
    data_retention_days: int = Field(default=7, alias="DATA_RETENTION_DAYS")
    
    # Security
    secret_key: str = Field(default="change-this-in-production", alias="SECRET_KEY")
    
    # Alert Channels
    enable_email: bool = Field(default=False, alias="ENABLE_EMAIL")
    enable_sms: bool = Field(default=False, alias="ENABLE_SMS")
    enable_webhook: bool = Field(default=False, alias="ENABLE_WEBHOOK")
    webhook_url: str = Field(default="", alias="WEBHOOK_URL")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Feature engineering configuration
FEATURE_CONFIG = {
    "torso_angle": {
        "description": "Angle between torso and vertical (degrees)",
        "min": 0.0,
        "max": 180.0,
        "critical_threshold": 30.0  # Fall if < 30 degrees
    },
    "bbox_ratio": {
        "description": "Bounding box height/width ratio",
        "min": 0.0,
        "max": 5.0,
        "critical_threshold": 1.0  # Fall if < 1.0
    },
    "com_velocity": {
        "description": "Center of mass Y-velocity (pixels/frame)",
        "min": -50.0,
        "max": 50.0,
        "critical_threshold": 10.0  # Fall if > 10
    },
    "hip_velocity": {
        "description": "Hip Y-velocity (pixels/frame)",
        "min": -50.0,
        "max": 50.0,
        "critical_threshold": 10.0
    },
    "head_velocity": {
        "description": "Head Y-velocity (pixels/frame)",
        "min": -50.0,
        "max": 50.0,
        "critical_threshold": 10.0
    },
    "joint_spread": {
        "description": "Maximum distance between joints (pixels)",
        "min": 0.0,
        "max": 1000.0,
        "critical_threshold": 100.0  # Fall if < 100
    },
    "head_hip_distance": {
        "description": "Distance from head to hip (pixels)",
        "min": 0.0,
        "max": 500.0,
        "critical_threshold": 50.0  # Fall if < 50
    }
}

# YOLO keypoint indices (COCO format)
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

# Dataset configuration
DATASET_CONFIG = {
    "ur_fall": {
        "name": "UR Fall Detection",
        "path": "./datasets/UR_Fall",
        "fall_folders": ["fall"],
        "non_fall_folders": ["adl"]  # Activities of Daily Living
    },
    "up_fall": {
        "name": "UP-Fall Detection",
        "path": "./datasets/UP_Fall",
        "fall_folders": ["Falls"],
        "non_fall_folders": ["ADLs"]
    },
    "le2i": {
        "name": "Le2i Fall Detection",
        "path": "./datasets/Le2i",
        "fall_folders": ["fall"],
        "non_fall_folders": ["not_fall"]
    }
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
    
    # Class balancing
    "use_class_weights": True,
    "fall_weight": 3.0,  # Falls are rare, weight them higher
    
    # Data augmentation
    "augmentation": {
        "temporal_jitter": 0.1,  # ±10% frame timing variation
        "speed_variation": (0.8, 1.2),  # 0.8x to 1.2x speed
        "pose_noise": 0.02,  # ±2% coordinate noise
        "horizontal_flip": 0.5  # 50% chance
    },
    
    # Target metrics
    "target_precision": 0.97,
    "target_recall": 0.95,
    "target_f1": 0.96
}
