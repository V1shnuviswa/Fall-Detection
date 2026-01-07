"""
Dataset loader and video processing pipeline.

This module handles:
1. Loading videos from multiple datasets
2. Sampling frames at target FPS
3. Organizing data for pose extraction
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import shutil

from ..utils.config import settings, DATASET_CONFIG
from ..utils.logger import setup_logger, LoggerMixin


logger = setup_logger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a video file."""
    video_path: Path
    dataset_name: str
    is_fall: bool
    original_fps: float
    total_frames: int
    duration: float
    label: str


class DatasetLoader(LoggerMixin):
    """
    Load and process videos from multiple fall detection datasets.
    
    Supports:
    - UR Fall Detection Dataset
    - UP-Fall Detection Dataset
    - Le2i Fall Detection Dataset
    """
    
    def __init__(
        self,
        dataset_root: Path = Path("./datasets"),
        output_root: Path = Path("./data/processed"),
        target_fps: int = 10
    ):
        """
        Initialize dataset loader.
        
        Args:
            dataset_root: Root directory containing all datasets
            output_root: Directory to save processed data
            target_fps: Target FPS for frame sampling (default: 10)
        """
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.target_fps = target_fps
        
        # Create output directories
        self.frames_dir = output_root / "frames"
        self.metadata_dir = output_root / "metadata"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DatasetLoader initialized with target FPS: {target_fps}")
    
    def discover_videos(self) -> List[VideoMetadata]:
        """
        Discover all videos from configured datasets.
        
        Returns:
            List of VideoMetadata objects
        """
        all_videos = []
        
        for dataset_id, config in DATASET_CONFIG.items():
            dataset_path = self.dataset_root / Path(config["path"]).name
            
            if not dataset_path.exists():
                self.logger.warning(f"Dataset not found: {dataset_path}")
                continue
            
            self.logger.info(f"Scanning dataset: {config['name']}")
            
            # Scan fall videos
            for fall_folder in config["fall_folders"]:
                folder_path = dataset_path / fall_folder
                if folder_path.exists():
                    videos = self._scan_folder(folder_path, dataset_id, is_fall=True)
                    all_videos.extend(videos)
                    self.logger.info(f"Found {len(videos)} fall videos in {fall_folder}")
            
            # Scan non-fall videos
            for non_fall_folder in config["non_fall_folders"]:
                folder_path = dataset_path / non_fall_folder
                if folder_path.exists():
                    videos = self._scan_folder(folder_path, dataset_id, is_fall=False)
                    all_videos.extend(videos)
                    self.logger.info(f"Found {len(videos)} non-fall videos in {non_fall_folder}")
        
        self.logger.info(f"Total videos discovered: {len(all_videos)}")
        fall_count = sum(1 for v in all_videos if v.is_fall)
        self.logger.info(f"Falls: {fall_count}, Non-falls: {len(all_videos) - fall_count}")
        
        return all_videos
    
    def _scan_folder(
        self,
        folder: Path,
        dataset_name: str,
        is_fall: bool
    ) -> List[VideoMetadata]:
        """Scan a folder for video files."""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
        videos = []
        
        for video_path in folder.rglob("*"):
            if video_path.suffix.lower() in video_extensions:
                try:
                    metadata = self._extract_video_metadata(
                        video_path, dataset_name, is_fall
                    )
                    if metadata:
                        videos.append(metadata)
                except Exception as e:
                    self.logger.error(f"Error processing {video_path}: {e}")
        
        return videos
    
    def _extract_video_metadata(
        self,
        video_path: Path,
        dataset_name: str,
        is_fall: bool
    ) -> Optional[VideoMetadata]:
        """Extract metadata from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.logger.warning(f"Cannot open video: {video_path}")
            return None
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoMetadata(
                video_path=video_path,
                dataset_name=dataset_name,
                is_fall=is_fall,
                original_fps=fps,
                total_frames=total_frames,
                duration=duration,
                label="fall" if is_fall else "non_fall"
            )
        finally:
            cap.release()
    
    def sample_frames(
        self,
        video_metadata: VideoMetadata,
        save_frames: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Sample frames from a video at target FPS.
        
        Args:
            video_metadata: Video metadata
            save_frames: Whether to save frames to disk (optional, for debugging)
        
        Returns:
            Tuple of (frames, timestamps)
        """
        cap = cv2.VideoCapture(str(video_metadata.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_metadata.video_path}")
        
        frames = []
        timestamps = []
        
        # Calculate frame interval for target FPS
        original_fps = video_metadata.original_fps
        frame_interval = max(1, int(original_fps / self.target_fps))
        
        frame_idx = 0
        saved_count = 0
        
        # Create output directory for this video if saving
        if save_frames:
            video_id = video_metadata.video_path.stem
            output_dir = self.frames_dir / video_metadata.dataset_name / video_metadata.label / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frame at target interval
                if frame_idx % frame_interval == 0:
                    frames.append(frame)
                    timestamp = frame_idx / original_fps
                    timestamps.append(timestamp)
                    
                    # Optionally save frame
                    if save_frames:
                        frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                    
                    saved_count += 1
                
                frame_idx += 1
        
        finally:
            cap.release()
        
        return frames, timestamps
    
    def process_all_videos(
        self,
        videos: Optional[List[VideoMetadata]] = None,
        save_frames: bool = False
    ) -> Dict[str, any]:
        """
        Process all videos and extract frames.
        
        Args:
            videos: List of videos to process (if None, discovers automatically)
            save_frames: Whether to save frames to disk
        
        Returns:
            Dictionary with processing statistics
        """
        if videos is None:
            videos = self.discover_videos()
        
        if not videos:
            raise ValueError("No videos found to process")
        
        stats = {
            "total_videos": len(videos),
            "processed": 0,
            "failed": 0,
            "total_frames": 0,
            "fall_videos": 0,
            "non_fall_videos": 0
        }
        
        # Process each video
        for video in tqdm(videos, desc="Processing videos"):
            try:
                frames, timestamps = self.sample_frames(video, save_frames=save_frames)
                
                # Save metadata
                metadata_file = self.metadata_dir / f"{video.video_path.stem}.json"
                metadata_dict = {
                    "video_path": str(video.video_path),
                    "dataset_name": video.dataset_name,
                    "is_fall": video.is_fall,
                    "label": video.label,
                    "original_fps": video.original_fps,
                    "sampled_fps": self.target_fps,
                    "total_frames": len(frames),
                    "duration": video.duration,
                    "timestamps": timestamps
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
                
                stats["processed"] += 1
                stats["total_frames"] += len(frames)
                if video.is_fall:
                    stats["fall_videos"] += 1
                else:
                    stats["non_fall_videos"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process {video.video_path}: {e}")
                stats["failed"] += 1
        
        # Save overall statistics
        stats_file = self.output_root / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Processing complete: {stats}")
        return stats


def main():
    """Main function to run dataset processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process fall detection datasets")
    parser.add_argument("--dataset-root", type=Path, default=Path("./datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("./data/processed"))
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--save-frames", action="store_true", help="Save frames to disk (large storage)")
    
    args = parser.parse_args()
    
    loader = DatasetLoader(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        target_fps=args.target_fps
    )
    
    stats = loader.process_all_videos(save_frames=args.save_frames)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total videos: {stats['total_videos']}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Fall videos: {stats['fall_videos']}")
    print(f"Non-fall videos: {stats['non_fall_videos']}")
    print("="*50)


if __name__ == "__main__":
    main()
