"""
YOLO Pose Extraction Module

Extracts 17 keypoints from frames using YOLOv8-Pose.
This is the spatial feature extraction layer.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from tqdm import tqdm
from ultralytics import YOLO

from ..utils.config import settings, KEYPOINT_INDICES
from ..utils.logger import setup_logger, LoggerMixin, log_performance


logger = setup_logger(__name__)


@dataclass
class PoseData:
    """Container for pose extraction results."""
    frame_idx: int
    timestamp: float
    keypoints: np.ndarray  # Shape: (17, 3) - [x, y, confidence]
    bbox: np.ndarray  # Shape: (4,) - [x1, y1, x2, y2]
    bbox_confidence: float
    person_id: int = 0  # For multi-person scenarios


class PoseExtractor(LoggerMixin):
    """
    Extract human poses from frames using YOLOv8-Pose.
    
    Key features:
    - Uses pre-trained YOLOv8m-pose (no fine-tuning needed)
    - Extracts 17 COCO keypoints
    - Handles multi-person detection (selects largest/most confident)
    - GPU acceleration
    """
    
    def __init__(
        self,
        model_name: str = "yolov8m-pose.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize pose extractor.
        
        Args:
            model_name: YOLOv8 pose model name
            device: Device to run inference on ("cuda" or "cpu")
            confidence_threshold: Minimum confidence for person detection
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and self.device == "cpu":
            self.logger.warning("CUDA requested but not available, using CPU")
        
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        self.logger.info(f"Loading {model_name} on {self.device}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.logger.info("Pose extractor initialized")
    
    @log_performance()
    def extract_pose(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0
    ) -> Optional[PoseData]:
        """
        Extract pose from a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            frame_idx: Frame index in video
            timestamp: Timestamp in seconds
        
        Returns:
            PoseData object or None if no person detected
        """
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        # Extract person detections with poses
        if results.keypoints is None or len(results.keypoints) == 0:
            return None
        
        # Get boxes and keypoints
        boxes = results.boxes
        keypoints = results.keypoints
        
        if len(boxes) == 0:
            return None
        
        # Select the most confident/largest person
        best_idx = self._select_best_person(boxes, keypoints)
        
        # Extract data for selected person
        bbox = boxes[best_idx].xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        bbox_conf = float(boxes[best_idx].conf[0])
        kpts = keypoints[best_idx].data[0].cpu().numpy()  # Shape: (17, 3)
        
        # Filter by confidence
        if bbox_conf < self.confidence_threshold:
            return None
        
        return PoseData(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=kpts,
            bbox=bbox,
            bbox_confidence=bbox_conf,
            person_id=0
        )
    
    def _select_best_person(self, boxes, keypoints) -> int:
        """
        Select the best person detection from multiple candidates.
        Strategy: Highest confidence * largest area
        """
        scores = []
        for i in range(len(boxes)):
            bbox = boxes[i].xyxy[0].cpu().numpy()
            conf = float(boxes[i].conf[0])
            
            # Calculate area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Combined score
            score = conf * area
            scores.append(score)
        
        return int(np.argmax(scores))
    
    def extract_from_video_metadata(
        self,
        metadata_path: Path,
        frames_dir: Optional[Path] = None,
        video_path: Optional[Path] = None
    ) -> List[PoseData]:
        """
        Extract poses from video using metadata file.
        
        Args:
            metadata_path: Path to video metadata JSON
            frames_dir: Directory with saved frames (if available)
            video_path: Path to video file (if frames not saved)
        
        Returns:
            List of PoseData objects
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        poses = []
        
        # Option 1: Use saved frames
        if frames_dir and frames_dir.exists():
            self.logger.info(f"Extracting poses from saved frames in {frames_dir}")
            frame_files = sorted(frames_dir.glob("*.jpg"))
            
            for i, frame_file in enumerate(tqdm(frame_files, desc="Extracting poses")):
                frame = cv2.imread(str(frame_file))
                timestamp = metadata["timestamps"][i] if i < len(metadata["timestamps"]) else 0.0
                
                pose = self.extract_pose(frame, frame_idx=i, timestamp=timestamp)
                if pose:
                    poses.append(pose)
        
        # Option 2: Read from video
        elif video_path:
            self.logger.info(f"Extracting poses from video {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = metadata["timestamps"][frame_idx] if frame_idx < len(metadata["timestamps"]) else 0.0
                pose = self.extract_pose(frame, frame_idx=frame_idx, timestamp=timestamp)
                if pose:
                    poses.append(pose)
                
                frame_idx += 1
            
            cap.release()
        
        else:
            raise ValueError("Either frames_dir or video_path must be provided")
        
        self.logger.info(f"Extracted {len(poses)} poses")
        return poses
    
    def save_poses(
        self,
        poses: List[PoseData],
        output_path: Path
    ):
        """Save extracted poses to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        poses_dict = {
            "total_frames": len(poses),
            "poses": []
        }
        
        for pose in poses:
            pose_dict = {
                "frame_idx": pose.frame_idx,
                "timestamp": pose.timestamp,
                "keypoints": pose.keypoints.tolist(),
                "bbox": pose.bbox.tolist(),
                "bbox_confidence": pose.bbox_confidence,
                "person_id": pose.person_id
            }
            poses_dict["poses"].append(pose_dict)
        
        with open(output_path, 'w') as f:
            json.dump(poses_dict, f)
        
        self.logger.info(f"Saved poses to {output_path}")
    
    @staticmethod
    def load_poses(poses_path: Path) -> List[PoseData]:
        """Load poses from saved file."""
        with open(poses_path, 'r') as f:
            data = json.load(f)
        
        poses = []
        for pose_dict in data["poses"]:
            pose = PoseData(
                frame_idx=pose_dict["frame_idx"],
                timestamp=pose_dict["timestamp"],
                keypoints=np.array(pose_dict["keypoints"]),
                bbox=np.array(pose_dict["bbox"]),
                bbox_confidence=pose_dict["bbox_confidence"],
                person_id=pose_dict["person_id"]
            )
            poses.append(pose)
        
        return poses
    
    def batch_process_dataset(
        self,
        metadata_dir: Path,
        output_dir: Path,
        dataset_root: Optional[Path] = None
    ):
        """
        Process all videos in a dataset.
        
        Args:
            metadata_dir: Directory with video metadata JSON files
            output_dir: Directory to save pose data
            dataset_root: Root directory of dataset (for video paths)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_files = list(metadata_dir.glob("*.json"))
        if not metadata_files:
            raise ValueError(f"No metadata files found in {metadata_dir}")
        
        self.logger.info(f"Processing {len(metadata_files)} videos")
        
        stats = {"processed": 0, "failed": 0, "total_poses": 0}
        
        for metadata_file in tqdm(metadata_files, desc="Processing videos"):
            try:
                # Load metadata to get video path
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                video_path = Path(metadata["video_path"])
                if not video_path.exists() and dataset_root:
                    video_path = dataset_root / video_path.name
                
                # Extract poses
                poses = self.extract_from_video_metadata(
                    metadata_path=metadata_file,
                    video_path=video_path
                )
                
                # Save poses
                output_file = output_dir / f"{metadata_file.stem}_poses.json"
                self.save_poses(poses, output_file)
                
                stats["processed"] += 1
                stats["total_poses"] += len(poses)
                
            except Exception as e:
                self.logger.error(f"Failed to process {metadata_file}: {e}")
                stats["failed"] += 1
        
        # Save statistics
        stats_file = output_dir / "pose_extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Batch processing complete: {stats}")


def main():
    """Main function to run pose extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract poses from fall detection videos")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default="yolov8m-pose.pt")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    extractor = PoseExtractor(
        model_name=args.model,
        device=args.device
    )
    
    extractor.batch_process_dataset(
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root
    )


if __name__ == "__main__":
    main()
