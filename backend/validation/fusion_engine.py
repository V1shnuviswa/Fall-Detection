"""
Confidence Fusion Engine

Combines multiple detection signals into final decision.

Formula:
final_score = 0.65 × temporal_model_score
            + 0.20 × posture_score
            + 0.15 × motion_score

Trigger fall only if:
- final_score ≥ 0.88
- Score maintained for ≥ 2 seconds
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
import time

from ..utils.config import settings
from ..utils.logger import setup_logger, LoggerMixin


logger = setup_logger(__name__)


@dataclass
class FusionResult:
    """Result of confidence fusion."""
    final_score: float
    component_scores: Dict[str, float]
    is_fall: bool
    confidence_level: str  # 'low', 'medium', 'high', 'critical'
    time_above_threshold: float
    ready_to_trigger: bool


class ConfidenceFusionEngine(LoggerMixin):
    """
    Fuse multiple confidence signals for robust fall detection.
    
    Components:
    1. Temporal model (LSTM): 65% weight
    2. Posture score: 20% weight
    3. Motion score: 15% weight
    """
    
    def __init__(
        self,
        temporal_weight: float = None,
        posture_weight: float = None,
        motion_weight: float = None,
        threshold: float = None,
        confirmation_seconds: float = None,
        fps: int = 10
    ):
        """
        Initialize fusion engine.
        
        Args:
            temporal_weight: Weight for LSTM model (default: 0.65)
            posture_weight: Weight for posture score (default: 0.20)
            motion_weight: Weight for motion score (default: 0.15)
            threshold: Final threshold for fall (default: 0.88)
            confirmation_seconds: Seconds to maintain score (default: 2)
            fps: Frames per second
        """
        self.temporal_weight = temporal_weight or settings.temporal_weight
        self.posture_weight = posture_weight or settings.posture_weight
        self.motion_weight = motion_weight or settings.motion_weight
        self.threshold = threshold or settings.fall_threshold
        self.confirmation_seconds = confirmation_seconds or settings.confirmation_seconds
        self.fps = fps
        
        # Validate weights sum to 1
        total_weight = self.temporal_weight + self.posture_weight + self.motion_weight
        if not np.isclose(total_weight, 1.0):
            self.logger.warning(f"Weights sum to {total_weight}, normalizing...")
            self.temporal_weight /= total_weight
            self.posture_weight /= total_weight
            self.motion_weight /= total_weight
        
        # Confirmation buffer
        confirmation_frames = int(self.confirmation_seconds * self.fps)
        self.score_history = deque(maxlen=confirmation_frames)
        
        # Tracking
        self.first_detection_time = None
        self.last_score = 0.0
        
        self.logger.info("Fusion engine initialized")
        self.logger.info(f"  Weights: temporal={self.temporal_weight:.2f}, "
                        f"posture={self.posture_weight:.2f}, motion={self.motion_weight:.2f}")
        self.logger.info(f"  Threshold: {self.threshold}")
        self.logger.info(f"  Confirmation: {self.confirmation_seconds}s")
    
    def compute_posture_score(self, features: Dict[str, float]) -> float:
        """
        Compute posture score from features.
        
        Indicators of fall posture:
        - Low torso angle (horizontal)
        - Low bbox ratio (wide)
        - Small head-hip distance
        
        Returns:
            Score in [0, 1]
        """
        scores = []
        
        # Torso angle: lower is worse (more horizontal)
        torso_angle = features.get('torso_angle', 90)
        torso_score = 1.0 - min(torso_angle / 90.0, 1.0)  # Inverted
        scores.append(torso_score)
        
        # Bbox ratio: lower is worse
        bbox_ratio = features.get('bbox_ratio', 2.0)
        bbox_score = 1.0 - min(bbox_ratio / 2.0, 1.0)  # Inverted
        scores.append(bbox_score)
        
        # Head-hip distance: smaller is worse
        head_hip_dist = features.get('head_hip_distance', 200)
        head_hip_score = 1.0 - min(head_hip_dist / 200.0, 1.0)  # Inverted
        scores.append(head_hip_score)
        
        # Joint spread: smaller is worse (collapsed)
        joint_spread = features.get('joint_spread', 300)
        spread_score = 1.0 - min(joint_spread / 300.0, 1.0)  # Inverted
        scores.append(spread_score)
        
        # Average all posture indicators
        posture_score = np.mean(scores)
        
        return float(np.clip(posture_score, 0.0, 1.0))
    
    def compute_motion_score(self, features: Dict[str, float]) -> float:
        """
        Compute motion score from features.
        
        Indicators of fall motion:
        - High downward velocity (COM, hip, head)
        - Rapid motion
        
        Returns:
            Score in [0, 1]
        """
        # Get velocities
        com_velocity = abs(features.get('com_velocity', 0))
        hip_velocity = abs(features.get('hip_velocity', 0))
        head_velocity = abs(features.get('head_velocity', 0))
        
        # Normalize to [0, 1] range
        # High velocity (>15) = score of 1.0
        velocity_threshold = 15.0
        
        com_score = min(com_velocity / velocity_threshold, 1.0)
        hip_score = min(hip_velocity / velocity_threshold, 1.0)
        head_score = min(head_velocity / velocity_threshold, 1.0)
        
        # Average velocities
        motion_score = np.mean([com_score, hip_score, head_score])
        
        return float(np.clip(motion_score, 0.0, 1.0))
    
    def fuse(
        self,
        temporal_score: float,
        features: Dict[str, float]
    ) -> FusionResult:
        """
        Fuse all confidence scores.
        
        Args:
            temporal_score: LSTM model output [0, 1]
            features: Feature dictionary with posture/motion info
        
        Returns:
            FusionResult object
        """
        # Compute component scores
        posture_score = self.compute_posture_score(features)
        motion_score = self.compute_motion_score(features)
        
        # Weighted fusion
        final_score = (
            self.temporal_weight * temporal_score +
            self.posture_weight * posture_score +
            self.motion_weight * motion_score
        )
        
        self.last_score = final_score
        
        # Add to history
        self.score_history.append(final_score)
        
        # Check if above threshold
        is_above_threshold = final_score >= self.threshold
        
        # Calculate time above threshold
        if is_above_threshold:
            if self.first_detection_time is None:
                self.first_detection_time = time.time()
            time_above = time.time() - self.first_detection_time
        else:
            self.first_detection_time = None
            time_above = 0.0
        
        # Check if confirmed (score maintained for required duration)
        ready_to_trigger = False
        if len(self.score_history) == self.score_history.maxlen:
            # Check if all recent scores above threshold
            recent_above = all(s >= self.threshold for s in self.score_history)
            if recent_above:
                ready_to_trigger = True
        
        # Determine confidence level
        if final_score >= 0.95:
            confidence_level = 'critical'
        elif final_score >= 0.90:
            confidence_level = 'high'
        elif final_score >= 0.80:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        result = FusionResult(
            final_score=float(final_score),
            component_scores={
                'temporal': float(temporal_score),
                'posture': float(posture_score),
                'motion': float(motion_score)
            },
            is_fall=is_above_threshold,
            confidence_level=confidence_level,
            time_above_threshold=time_above,
            ready_to_trigger=ready_to_trigger
        )
        
        # Log high-confidence detections
        if final_score >= self.threshold:
            self.logger.info(
                f"High confidence detection: {final_score:.3f} "
                f"(T:{temporal_score:.2f}, P:{posture_score:.2f}, M:{motion_score:.2f}) "
                f"- Time: {time_above:.1f}s - Ready: {ready_to_trigger}"
            )
        
        return result
    
    def reset(self):
        """Reset fusion engine state."""
        self.score_history.clear()
        self.first_detection_time = None
        self.last_score = 0.0
        self.logger.debug("Fusion engine reset")
    
    def get_state(self) -> Dict:
        """Get current fusion state."""
        return {
            'last_score': self.last_score,
            'score_history_length': len(self.score_history),
            'time_above_threshold': (
                time.time() - self.first_detection_time
                if self.first_detection_time else 0.0
            ),
            'threshold': self.threshold
        }


def create_fusion_engine() -> ConfidenceFusionEngine:
    """Create fusion engine with default settings."""
    return ConfidenceFusionEngine(
        temporal_weight=settings.temporal_weight,
        posture_weight=settings.posture_weight,
        motion_weight=settings.motion_weight,
        threshold=settings.fall_threshold,
        confirmation_seconds=settings.confirmation_seconds,
        fps=settings.target_fps
    )


if __name__ == "__main__":
    # Test fusion engine
    engine = create_fusion_engine()
    
    print("="*60)
    print("CONFIDENCE FUSION ENGINE TEST")
    print("="*60)
    
    # Simulate fall detection over time
    print("\nSimulating fall detection sequence:")
    print("-"*60)
    
    # Test features for a fall
    fall_features = {
        'torso_angle': 20.0,
        'bbox_ratio': 0.7,
        'com_velocity': 12.0,
        'hip_velocity': 10.0,
        'head_velocity': 11.0,
        'head_hip_distance': 50.0,
        'joint_spread': 80.0
    }
    
    # Simulate multiple frames
    for frame in range(25):
        temporal_score = 0.90 + frame * 0.002  # Gradually increasing
        
        result = engine.fuse(temporal_score, fall_features)
        
        if frame % 5 == 0:  # Print every 5 frames
            print(f"Frame {frame}:")
            print(f"  Final score: {result.final_score:.3f}")
            print(f"  Components: T={result.component_scores['temporal']:.2f}, "
                  f"P={result.component_scores['posture']:.2f}, "
                  f"M={result.component_scores['motion']:.2f}")
            print(f"  Confidence: {result.confidence_level}")
            print(f"  Time above threshold: {result.time_above_threshold:.1f}s")
            print(f"  Ready to trigger: {result.ready_to_trigger}")
            print()
    
    print("="*60)
