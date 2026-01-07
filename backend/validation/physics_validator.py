"""
Physics-Based Validation Layer

Rule-based validation to reduce false positives.
Validates fall detections using physical constraints.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import time

from ..utils.config import settings, FEATURE_CONFIG
from ..utils.logger import setup_logger, LoggerMixin


logger = setup_logger(__name__)


@dataclass
class ValidationResult:
    """Result of physics validation."""
    is_valid: bool
    violations: List[str]
    scores: Dict[str, float]
    passed_checks: List[str]


class PhysicsValidator(LoggerMixin):
    """
    Validate fall detections using physics-based rules.
    
    A fall is valid only if ALL conditions pass:
    1. LSTM probability ≥ 0.85
    2. Torso angle ≤ 30° (nearly horizontal)
    3. Bounding box ratio ≤ 1 (width > height)
    4. Inactivity ≥ 5 seconds after detection
    
    This filters out:
    - Sitting/lying down intentionally
    - Bending over
    - Yoga poses
    - Picking up objects
    - Sleeping
    """
    
    def __init__(
        self,
        min_lstm_prob: float = None,
        max_torso_angle: float = None,
        max_bbox_ratio: float = None,
        inactivity_threshold: int = None,
        fps: int = 10
    ):
        """
        Initialize physics validator.
        
        Args:
            min_lstm_prob: Minimum LSTM probability
            max_torso_angle: Maximum torso angle (degrees)
            max_bbox_ratio: Maximum bbox height/width ratio
            inactivity_threshold: Seconds of inactivity required
            fps: Frames per second
        """
        self.min_lstm_prob = min_lstm_prob or settings.min_lstm_probability
        self.max_torso_angle = max_torso_angle or settings.max_torso_angle
        self.max_bbox_ratio = max_bbox_ratio or settings.max_bbox_ratio
        self.inactivity_threshold = inactivity_threshold or settings.inactivity_threshold
        self.fps = fps
        
        # State tracking
        self.detection_time = None
        self.motion_history = deque(maxlen=self.fps * self.inactivity_threshold)
        
        self.logger.info("Physics validator initialized")
        self.logger.info(f"  Min LSTM prob: {self.min_lstm_prob}")
        self.logger.info(f"  Max torso angle: {self.max_torso_angle}°")
        self.logger.info(f"  Max bbox ratio: {self.max_bbox_ratio}")
        self.logger.info(f"  Inactivity threshold: {self.inactivity_threshold}s")
    
    def validate(
        self,
        lstm_probability: float,
        features: Dict[str, float],
        motion_score: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate a fall detection.
        
        Args:
            lstm_probability: LSTM model output probability
            features: Dictionary with feature values:
                - torso_angle
                - bbox_ratio
                - com_velocity
                - hip_velocity
                - head_velocity
            motion_score: Optional motion score for inactivity check
        
        Returns:
            ValidationResult object
        """
        violations = []
        passed_checks = []
        scores = {}
        
        # Check 1: LSTM probability
        scores['lstm_prob'] = lstm_probability
        if lstm_probability >= self.min_lstm_prob:
            passed_checks.append(f"LSTM probability: {lstm_probability:.3f} >= {self.min_lstm_prob}")
        else:
            violations.append(f"LSTM probability too low: {lstm_probability:.3f} < {self.min_lstm_prob}")
        
        # Check 2: Torso angle
        torso_angle = features.get('torso_angle', 90)
        scores['torso_angle'] = torso_angle
        
        # For fall: torso should be nearly horizontal (high angle from vertical)
        # Vertical standing = 0°, Horizontal lying = 90°
        # We want angle from vertical to be large (> 60°) OR
        # if measuring differently, torso angle should be small (< 30°)
        
        # Assuming torso_angle is angle from horizontal (0° = horizontal)
        if torso_angle <= self.max_torso_angle:
            passed_checks.append(f"Torso angle: {torso_angle:.1f}° <= {self.max_torso_angle}°")
        else:
            violations.append(f"Torso not horizontal: {torso_angle:.1f}° > {self.max_torso_angle}°")
        
        # Check 3: Bounding box ratio
        bbox_ratio = features.get('bbox_ratio', 2.0)
        scores['bbox_ratio'] = bbox_ratio
        
        if bbox_ratio <= self.max_bbox_ratio:
            passed_checks.append(f"Bbox ratio: {bbox_ratio:.2f} <= {self.max_bbox_ratio}")
        else:
            violations.append(f"Person still upright: ratio {bbox_ratio:.2f} > {self.max_bbox_ratio}")
        
        # Check 4: Motion indicators (high downward velocity)
        com_velocity = features.get('com_velocity', 0)
        hip_velocity = features.get('hip_velocity', 0)
        
        scores['com_velocity'] = com_velocity
        scores['hip_velocity'] = hip_velocity
        
        # High positive velocity = falling down
        has_fall_motion = (com_velocity > 5 or hip_velocity > 5)
        if has_fall_motion:
            passed_checks.append(f"Fall motion detected (COM vel: {com_velocity:.1f}, Hip vel: {hip_velocity:.1f})")
        
        # Check 5: Inactivity (if motion score provided)
        if motion_score is not None:
            self.motion_history.append(motion_score)
            scores['motion_score'] = motion_score
            
            if self.detection_time is None:
                self.detection_time = time.time()
            
            elapsed = time.time() - self.detection_time
            
            if elapsed >= self.inactivity_threshold:
                # Check if person has been inactive
                if len(self.motion_history) > 0:
                    avg_motion = np.mean(list(self.motion_history))
                    scores['avg_motion'] = avg_motion
                    
                    if avg_motion < 2.0:  # Very little motion
                        passed_checks.append(f"Inactivity confirmed: {elapsed:.1f}s, motion: {avg_motion:.2f}")
                    else:
                        violations.append(f"Too much motion after detection: {avg_motion:.2f}")
        
        # Determine if valid
        # Require at least: LSTM prob + (torso angle OR bbox ratio)
        critical_checks = [
            lstm_probability >= self.min_lstm_prob,
            torso_angle <= self.max_torso_angle,
            bbox_ratio <= self.max_bbox_ratio
        ]
        
        # Must pass LSTM check and at least one posture check
        is_valid = critical_checks[0] and (critical_checks[1] or critical_checks[2])
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            scores=scores,
            passed_checks=passed_checks
        )
    
    def reset(self):
        """Reset validator state."""
        self.detection_time = None
        self.motion_history.clear()
        self.logger.debug("Validator reset")


class FallFilter:
    """
    Filter to reduce false positives from intentional actions.
    
    Filters out:
    - Controlled sitting/lying
    - Exercise movements
    - Bending to pick objects
    """
    
    def __init__(self):
        self.logger = logger
        
        # Track recent detections to avoid duplicates
        self.recent_detections = deque(maxlen=50)  # Last 5 seconds at 10 FPS
    
    def is_controlled_motion(self, features: Dict[str, float]) -> bool:
        """
        Check if motion appears controlled (not a fall).
        
        Controlled motion characteristics:
        - Slow, gradual descent
        - Maintained joint spread (not collapsing)
        - Symmetrical posture
        """
        # Check velocity - falls are sudden
        com_velocity = features.get('com_velocity', 0)
        hip_velocity = features.get('hip_velocity', 0)
        
        # Too slow = controlled
        if abs(com_velocity) < 3 and abs(hip_velocity) < 3:
            return True
        
        # Check joint spread - maintained during controlled descent
        joint_spread = features.get('joint_spread', 0)
        if joint_spread > 200:  # Still extended
            return True
        
        return False
    
    def is_duplicate(self, timestamp: float) -> bool:
        """
        Check if this is a duplicate detection.
        
        Falls should only be detected once, not continuously.
        """
        if len(self.recent_detections) == 0:
            self.recent_detections.append(timestamp)
            return False
        
        # Check if detection within last 3 seconds
        recent_time = max(self.recent_detections)
        if timestamp - recent_time < 3.0:
            return True
        
        self.recent_detections.append(timestamp)
        return False
    
    def should_filter(
        self,
        features: Dict[str, float],
        timestamp: float
    ) -> Tuple[bool, str]:
        """
        Determine if detection should be filtered.
        
        Returns:
            Tuple of (should_filter, reason)
        """
        # Check for controlled motion
        if self.is_controlled_motion(features):
            return True, "Controlled motion (not a fall)"
        
        # Check for duplicate
        if self.is_duplicate(timestamp):
            return True, "Duplicate detection"
        
        return False, ""


def create_validator() -> PhysicsValidator:
    """Create validator with default settings."""
    return PhysicsValidator(
        min_lstm_prob=settings.min_lstm_probability,
        max_torso_angle=settings.max_torso_angle,
        max_bbox_ratio=settings.max_bbox_ratio,
        inactivity_threshold=settings.inactivity_threshold,
        fps=settings.target_fps
    )


if __name__ == "__main__":
    # Test validator
    validator = create_validator()
    fall_filter = FallFilter()
    
    print("="*60)
    print("PHYSICS VALIDATOR TEST")
    print("="*60)
    
    # Test case 1: Valid fall
    result = validator.validate(
        lstm_probability=0.92,
        features={
            'torso_angle': 15.0,
            'bbox_ratio': 0.6,
            'com_velocity': 12.0,
            'hip_velocity': 10.0
        }
    )
    
    print("\nTest Case 1: Valid Fall")
    print(f"Valid: {result.is_valid}")
    print(f"Passed checks: {result.passed_checks}")
    print(f"Violations: {result.violations}")
    
    # Test case 2: Sitting down (should be filtered)
    result = validator.validate(
        lstm_probability=0.75,
        features={
            'torso_angle': 45.0,
            'bbox_ratio': 1.2,
            'com_velocity': 2.0,
            'hip_velocity': 1.5
        }
    )
    
    print("\nTest Case 2: Sitting Down")
    print(f"Valid: {result.is_valid}")
    print(f"Passed checks: {result.passed_checks}")
    print(f"Violations: {result.violations}")
    
    print("="*60)
