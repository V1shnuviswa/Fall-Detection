"""
Feature Engineering Pipeline

Converts raw pose data into 7-dimensional feature vectors for fall detection.

Features:
1. Torso angle
2. Bounding box ratio (height/width)
3. Center of mass Y-velocity
4. Hip Y-velocity
5. Head Y-velocity
6. Joint spread
7. Head-to-hip distance
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm

from ..utils.config import KEYPOINT_INDICES, FEATURE_CONFIG
from ..utils.logger import setup_logger, LoggerMixin
from .pose_extractor import PoseData


logger = setup_logger(__name__)


@dataclass
class FeatureVector:
    """7-dimensional feature vector for a single frame."""
    frame_idx: int
    timestamp: float
    features: np.ndarray  # Shape: (7,)
    
    # Individual features for debugging
    torso_angle: float
    bbox_ratio: float
    com_velocity: float
    hip_velocity: float
    head_velocity: float
    joint_spread: float
    head_hip_distance: float


class FeatureEngineer(LoggerMixin):
    """
    Engineer features from pose sequences for fall detection.
    
    This is the most critical component - features must capture fall dynamics.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.logger.info("Feature engineer initialized")
    
    def extract_features_from_poses(
        self,
        poses: List[PoseData],
        window_size: int = 5
    ) -> List[FeatureVector]:
        """
        Extract feature vectors from pose sequence.
        
        Args:
            poses: List of PoseData objects
            window_size: Window for velocity calculation
        
        Returns:
            List of FeatureVector objects
        """
        if len(poses) < window_size:
            self.logger.warning(f"Pose sequence too short: {len(poses)} < {window_size}")
            return []
        
        features = []
        
        for i in range(len(poses)):
            try:
                feature = self._compute_frame_features(poses, i, window_size)
                if feature:
                    features.append(feature)
            except Exception as e:
                self.logger.debug(f"Failed to compute features for frame {i}: {e}")
        
        return features
    
    def _compute_frame_features(
        self,
        poses: List[PoseData],
        idx: int,
        window_size: int
    ) -> Optional[FeatureVector]:
        """Compute all features for a single frame."""
        current_pose = poses[idx]
        kpts = current_pose.keypoints
        
        # Check if enough keypoints are visible
        visible_count = np.sum(kpts[:, 2] > 0.3)  # Confidence > 0.3
        if visible_count < 8:  # Need at least 8 visible keypoints
            return None
        
        # 1. Torso angle
        torso_angle = self._compute_torso_angle(kpts)
        
        # 2. Bounding box ratio
        bbox_ratio = self._compute_bbox_ratio(current_pose.bbox)
        
        # 3. Center of mass velocity
        com_velocity = self._compute_velocity(
            poses, idx, window_size, self._get_center_of_mass
        )
        
        # 4. Hip velocity
        hip_velocity = self._compute_velocity(
            poses, idx, window_size, self._get_hip_position
        )
        
        # 5. Head velocity
        head_velocity = self._compute_velocity(
            poses, idx, window_size, self._get_head_position
        )
        
        # 6. Joint spread
        joint_spread = self._compute_joint_spread(kpts)
        
        # 7. Head-to-hip distance
        head_hip_dist = self._compute_head_hip_distance(kpts)
        
        # Combine into feature vector
        features = np.array([
            torso_angle,
            bbox_ratio,
            com_velocity,
            hip_velocity,
            head_velocity,
            joint_spread,
            head_hip_dist
        ], dtype=np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return FeatureVector(
            frame_idx=current_pose.frame_idx,
            timestamp=current_pose.timestamp,
            features=features,
            torso_angle=torso_angle,
            bbox_ratio=bbox_ratio,
            com_velocity=com_velocity,
            hip_velocity=hip_velocity,
            head_velocity=head_velocity,
            joint_spread=joint_spread,
            head_hip_distance=head_hip_dist
        )
    
    def _compute_torso_angle(self, kpts: np.ndarray) -> float:
        """
        Compute angle between torso and vertical.
        
        Torso is defined as vector from shoulder midpoint to hip midpoint.
        Vertical fall: angle approaches 90 degrees (horizontal)
        Standing: angle approaches 0 degrees (vertical)
        """
        # Get shoulder and hip positions
        left_shoulder = kpts[KEYPOINT_INDICES["left_shoulder"]][:2]
        right_shoulder = kpts[KEYPOINT_INDICES["right_shoulder"]][:2]
        left_hip = kpts[KEYPOINT_INDICES["left_hip"]][:2]
        right_hip = kpts[KEYPOINT_INDICES["right_hip"]][:2]
        
        # Check visibility
        if (kpts[KEYPOINT_INDICES["left_shoulder"]][2] < 0.3 or
            kpts[KEYPOINT_INDICES["right_shoulder"]][2] < 0.3 or
            kpts[KEYPOINT_INDICES["left_hip"]][2] < 0.3 or
            kpts[KEYPOINT_INDICES["right_hip"]][2] < 0.3):
            return 90.0  # Assume worst case if not visible
        
        # Compute midpoints
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        
        # Torso vector (from shoulders to hips)
        torso_vector = hip_mid - shoulder_mid
        
        # Vertical vector (pointing down)
        vertical_vector = np.array([0, 1])
        
        # Compute angle
        dot_product = np.dot(torso_vector, vertical_vector)
        torso_length = np.linalg.norm(torso_vector)
        
        if torso_length < 1e-6:
            return 90.0
        
        cos_angle = dot_product / torso_length
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return float(angle)
    
    def _compute_bbox_ratio(self, bbox: np.ndarray) -> float:
        """
        Compute bounding box height/width ratio.
        
        Standing: height > width (ratio > 1)
        Fallen: width > height (ratio < 1)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width < 1:
            return 5.0  # Max ratio
        
        ratio = height / width
        return float(np.clip(ratio, 0.0, 5.0))
    
    def _get_center_of_mass(self, kpts: np.ndarray) -> np.ndarray:
        """Compute center of mass from keypoints."""
        # Use visible keypoints only
        visible_mask = kpts[:, 2] > 0.3
        visible_kpts = kpts[visible_mask][:, :2]
        
        if len(visible_kpts) == 0:
            return np.array([0.0, 0.0])
        
        return np.mean(visible_kpts, axis=0)
    
    def _get_hip_position(self, kpts: np.ndarray) -> np.ndarray:
        """Get hip midpoint."""
        left_hip = kpts[KEYPOINT_INDICES["left_hip"]]
        right_hip = kpts[KEYPOINT_INDICES["right_hip"]]
        
        # Check visibility
        if left_hip[2] > 0.3 and right_hip[2] > 0.3:
            return (left_hip[:2] + right_hip[:2]) / 2
        elif left_hip[2] > 0.3:
            return left_hip[:2]
        elif right_hip[2] > 0.3:
            return right_hip[:2]
        else:
            return np.array([0.0, 0.0])
    
    def _get_head_position(self, kpts: np.ndarray) -> np.ndarray:
        """Get head position (nose or average of eyes/ears)."""
        nose = kpts[KEYPOINT_INDICES["nose"]]
        
        if nose[2] > 0.3:
            return nose[:2]
        
        # Fallback: average eyes and ears
        head_kpts = [
            kpts[KEYPOINT_INDICES["left_eye"]],
            kpts[KEYPOINT_INDICES["right_eye"]],
            kpts[KEYPOINT_INDICES["left_ear"]],
            kpts[KEYPOINT_INDICES["right_ear"]]
        ]
        
        visible_head = [kpt[:2] for kpt in head_kpts if kpt[2] > 0.3]
        if visible_head:
            return np.mean(visible_head, axis=0)
        
        return np.array([0.0, 0.0])
    
    def _compute_velocity(
        self,
        poses: List[PoseData],
        idx: int,
        window_size: int,
        position_func
    ) -> float:
        """
        Compute Y-velocity using finite differences.
        
        Positive velocity = downward motion (fall indicator)
        """
        if idx < window_size:
            return 0.0
        
        # Get positions at current and previous frames
        current_pos = position_func(poses[idx].keypoints)
        prev_pos = position_func(poses[idx - window_size].keypoints)
        
        # Time difference
        dt = poses[idx].timestamp - poses[idx - window_size].timestamp
        if dt < 1e-6:
            return 0.0
        
        # Y-velocity (positive = downward)
        velocity = (current_pos[1] - prev_pos[1]) / dt
        
        return float(np.clip(velocity, -50.0, 50.0))
    
    def _compute_joint_spread(self, kpts: np.ndarray) -> float:
        """
        Compute maximum distance between any two joints.
        
        Standing/moving: large spread
        Fallen/collapsed: small spread
        """
        # Get visible keypoints
        visible_mask = kpts[:, 2] > 0.3
        visible_kpts = kpts[visible_mask][:, :2]
        
        if len(visible_kpts) < 2:
            return 0.0
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(visible_kpts, metric='euclidean')
        
        max_spread = float(np.max(distances))
        return np.clip(max_spread, 0.0, 1000.0)
    
    def _compute_head_hip_distance(self, kpts: np.ndarray) -> float:
        """
        Compute vertical distance from head to hip.
        
        Standing: large distance
        Fallen: small distance
        """
        head_pos = self._get_head_position(kpts)
        hip_pos = self._get_hip_position(kpts)
        
        # Vertical distance
        distance = float(np.abs(head_pos[1] - hip_pos[1]))
        return np.clip(distance, 0.0, 500.0)
    
    def batch_process_poses(
        self,
        poses_dir: Path,
        output_dir: Path
    ):
        """
        Process all pose files in a directory.
        
        Args:
            poses_dir: Directory with pose JSON files
            output_dir: Directory to save feature data
        """
        from .pose_extractor import PoseExtractor
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pose_files = list(poses_dir.glob("*_poses.json"))
        if not pose_files:
            raise ValueError(f"No pose files found in {poses_dir}")
        
        self.logger.info(f"Processing {len(pose_files)} pose files")
        
        stats = {"processed": 0, "failed": 0, "total_features": 0}
        
        for pose_file in tqdm(pose_files, desc="Extracting features"):
            try:
                # Load poses
                poses = PoseExtractor.load_poses(pose_file)
                
                # Extract features
                features = self.extract_features_from_poses(poses)
                
                # Save features
                output_file = output_dir / pose_file.name.replace("_poses.json", "_features.json")
                self._save_features(features, output_file)
                
                stats["processed"] += 1
                stats["total_features"] += len(features)
                
            except Exception as e:
                self.logger.error(f"Failed to process {pose_file}: {e}")
                stats["failed"] += 1
        
        # Save statistics
        stats_file = output_dir / "feature_extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Batch processing complete: {stats}")
    
    def _save_features(self, features: List[FeatureVector], output_path: Path):
        """Save feature vectors to file."""
        data = {
            "total_frames": len(features),
            "feature_dim": 7,
            "features": []
        }
        
        for feat in features:
            feat_dict = {
                "frame_idx": feat.frame_idx,
                "timestamp": feat.timestamp,
                "vector": feat.features.tolist(),
                "components": {
                    "torso_angle": feat.torso_angle,
                    "bbox_ratio": feat.bbox_ratio,
                    "com_velocity": feat.com_velocity,
                    "hip_velocity": feat.hip_velocity,
                    "head_velocity": feat.head_velocity,
                    "joint_spread": feat.joint_spread,
                    "head_hip_distance": feat.head_hip_distance
                }
            }
            data["features"].append(feat_dict)
        
        with open(output_path, 'w') as f:
            json.dump(data, f)


def main():
    """Main function to run feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from pose data")
    parser.add_argument("--poses-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    
    args = parser.parse_args()
    
    engineer = FeatureEngineer()
    engineer.batch_process_poses(
        poses_dir=args.poses_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
