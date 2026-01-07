"""
Session Manager for WebSocket Connections

Manages user sessions and their detection states.
"""

import uuid
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from ..models.inference import RealTimeFallDetector
from ..data_processing.pose_extractor import PoseExtractor
from ..validation.physics_validator import PhysicsValidator, FallFilter
from ..validation.fusion_engine import ConfidenceFusionEngine
from ..utils.logger import setup_logger, LoggerMixin


logger = setup_logger(__name__)


@dataclass
class UserSession:
    """User session state."""
    session_id: str
    detector: RealTimeFallDetector
    pose_extractor: PoseExtractor
    validator: PhysicsValidator
    fusion_engine: ConfidenceFusionEngine
    fall_filter: FallFilter
    
    # State
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    fall_count: int = 0
    alert_triggered: bool = False
    alert_timestamp: Optional[datetime] = None
    
    # Statistics
    frames_processed: int = 0
    detections_count: int = 0


class SessionManager(LoggerMixin):
    """
    Manage active user sessions.
    
    Each session has its own:
    - Fall detector
    - Pose extractor
    - Validation pipeline
    """
    
    def __init__(
        self,
        model_path,
        device: str = 'cuda'
    ):
        """
        Initialize session manager.
        
        Args:
            model_path: Path to trained LSTM model
            device: Device for inference
        """
        self.model_path = model_path
        self.device = device
        self.sessions: Dict[str, UserSession] = {}
        
        self.logger.info("Session manager initialized")
    
    def create_session(self) -> str:
        """
        Create a new user session.
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        # Create detector
        detector = RealTimeFallDetector(
            model_path=self.model_path,
            device=self.device
        )
        
        # Create pose extractor
        pose_extractor = PoseExtractor(
            model_name="yolov8m-pose.pt",
            device=self.device
        )
        
        # Create validator
        validator = PhysicsValidator()
        
        # Create fusion engine
        fusion_engine = ConfidenceFusionEngine()
        
        # Create filter
        fall_filter = FallFilter()
        
        # Create session
        session = UserSession(
            session_id=session_id,
            detector=detector,
            pose_extractor=pose_extractor,
            validator=validator,
            fusion_engine=fusion_engine,
            fall_filter=fall_filter
        )
        
        self.sessions[session_id] = session
        
        self.logger.info(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def update_activity(self, session_id: str):
        """Update last activity timestamp."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Removed session: {session_id}")
    
    def cleanup_inactive_sessions(self, timeout_minutes: int = 30):
        """Remove inactive sessions."""
        now = datetime.now()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            inactive_time = (now - session.last_activity).total_seconds() / 60
            if inactive_time > timeout_minutes:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            self.remove_session(session_id)
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} inactive sessions")
    
    def get_stats(self) -> Dict:
        """Get manager statistics."""
        return {
            'active_sessions': len(self.sessions),
            'total_frames': sum(s.frames_processed for s in self.sessions.values()),
            'total_detections': sum(s.detections_count for s in self.sessions.values()),
            'total_falls': sum(s.fall_count for s in self.sessions.values())
        }
