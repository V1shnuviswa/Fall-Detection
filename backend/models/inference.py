"""
Real-time Inference Module

Handles real-time fall detection inference with temporal buffering.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import deque
import pickle

from .lstm_model import FallDetectionLSTM
from ..data_processing.feature_engineer import FeatureEngineer
from ..data_processing.pose_extractor import PoseData
from ..utils.config import settings
from ..utils.logger import setup_logger, LoggerMixin, log_performance


logger = setup_logger(__name__)


class RealTimeFallDetector(LoggerMixin):
    """
    Real-time fall detector with temporal buffering.
    
    Maintains a sliding window of features and runs inference
    when buffer is full.
    """
    
    def __init__(
        self,
        model_path: Path,
        device: str = 'cuda',
        buffer_size: int = 30,
        threshold: float = 0.85
    ):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            buffer_size: Size of temporal buffer (frames)
            threshold: Detection threshold
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.buffer_size = buffer_size
        self.threshold = threshold
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load normalization statistics
        norm_stats_path = model_path.parent / "normalization_stats.pkl"
        if norm_stats_path.exists():
            with open(norm_stats_path, 'rb') as f:
                norm_stats = pickle.load(f)
                self.mean = norm_stats['mean']
                self.std = norm_stats['std']
        else:
            self.logger.warning("Normalization stats not found, using defaults")
            self.mean = np.zeros((1, 1, 7))
            self.std = np.ones((1, 1, 7))
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Temporal buffer
        self.feature_buffer = deque(maxlen=buffer_size)
        self.pose_buffer = deque(maxlen=5)  # For velocity calculation
        
        # State tracking
        self.frame_count = 0
        self.last_prediction = 0.0
        
        self.logger.info(f"Real-time detector initialized on {self.device}")
    
    def _load_model(self, model_path: Path) -> FallDetectionLSTM:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model config
        model_config = checkpoint.get('model_config', {})
        
        # Create model
        model = FallDetectionLSTM(
            input_dim=model_config.get('input_dim', 7),
            hidden_dim1=model_config.get('hidden_dim1', 128),
            hidden_dim2=model_config.get('hidden_dim2', 64),
            bidirectional=model_config.get('bidirectional', False)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        self.logger.info(f"Model loaded successfully")
        return model
    
    @log_performance()
    def process_pose(self, pose: PoseData) -> Dict:
        """
        Process a single pose and update buffer.
        
        Args:
            pose: Pose data from current frame
        
        Returns:
            Detection result dictionary
        """
        self.frame_count += 1
        
        # Add to pose buffer
        self.pose_buffer.append(pose)
        
        # Need at least 5 poses for velocity calculation
        if len(self.pose_buffer) < 5:
            return {
                'ready': False,
                'probability': 0.0,
                'is_fall': False,
                'buffer_filled': len(self.feature_buffer) / self.buffer_size
            }
        
        # Extract features
        try:
            feature = self.feature_engineer._compute_frame_features(
                poses=list(self.pose_buffer),
                idx=len(self.pose_buffer) - 1,
                window_size=5
            )
            
            if feature is None:
                return {
                    'ready': False,
                    'probability': 0.0,
                    'is_fall': False,
                    'error': 'Failed to extract features'
                }
            
            # Add to feature buffer
            self.feature_buffer.append(feature.features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return {
                'ready': False,
                'probability': 0.0,
                'is_fall': False,
                'error': str(e)
            }
        
        # Check if buffer is full
        if len(self.feature_buffer) < self.buffer_size:
            return {
                'ready': False,
                'probability': self.last_prediction,
                'is_fall': False,
                'buffer_filled': len(self.feature_buffer) / self.buffer_size
            }
        
        # Run inference
        probability = self._predict()
        self.last_prediction = probability
        
        result = {
            'ready': True,
            'probability': float(probability),
            'is_fall': probability >= self.threshold,
            'buffer_filled': 1.0,
            'frame_count': self.frame_count
        }
        
        return result
    
    @log_performance()
    def _predict(self) -> float:
        """
        Run LSTM inference on current buffer.
        
        Returns:
            Fall probability [0, 1]
        """
        # Convert buffer to array
        features = np.array(list(self.feature_buffer))  # Shape: (30, 7)
        
        # Normalize
        features = (features - self.mean[0]) / self.std[0]
        
        # Add batch dimension
        features = features[np.newaxis, ...]  # Shape: (1, 30, 7)
        
        # Convert to tensor
        x = torch.from_numpy(features).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(x)
        
        probability = float(output.item())
        
        return probability
    
    def reset(self):
        """Reset buffers and state."""
        self.feature_buffer.clear()
        self.pose_buffer.clear()
        self.frame_count = 0
        self.last_prediction = 0.0
        self.logger.info("Detector reset")
    
    def get_state(self) -> Dict:
        """Get current detector state."""
        return {
            'frame_count': self.frame_count,
            'buffer_size': len(self.feature_buffer),
            'buffer_capacity': self.buffer_size,
            'buffer_filled': len(self.feature_buffer) / self.buffer_size,
            'last_prediction': self.last_prediction,
            'threshold': self.threshold
        }


class BatchInferenceEngine(LoggerMixin):
    """Batch inference for processing recorded videos."""
    
    def __init__(
        self,
        model_path: Path,
        device: str = 'cuda'
    ):
        """
        Initialize batch inference engine.
        
        Args:
            model_path: Path to trained model
            device: Device to run on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('model_config', {})
        
        self.model = FallDetectionLSTM(
            input_dim=model_config.get('input_dim', 7),
            hidden_dim1=model_config.get('hidden_dim1', 128),
            hidden_dim2=model_config.get('hidden_dim2', 64),
            bidirectional=model_config.get('bidirectional', False)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization stats
        norm_stats_path = model_path.parent / "normalization_stats.pkl"
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
            self.mean = norm_stats['mean']
            self.std = norm_stats['std']
        
        self.logger.info("Batch inference engine initialized")
    
    def predict_batch(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Predict on batch of feature sequences.
        
        Args:
            features: Array of shape (N, time_steps, feature_dim)
        
        Returns:
            Predictions of shape (N,)
        """
        # Normalize
        features = (features - self.mean) / self.std
        
        # Convert to tensor
        x = torch.from_numpy(features).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(x)
        
        predictions = outputs.cpu().numpy().flatten()
        
        return predictions


def load_for_inference(
    model_path: Path = None,
    device: str = 'cuda'
) -> RealTimeFallDetector:
    """
    Convenience function to load model for inference.
    
    Args:
        model_path: Path to model checkpoint (uses default if None)
        device: Device to run on
    
    Returns:
        Initialized detector
    """
    if model_path is None:
        model_path = settings.model_path
    
    detector = RealTimeFallDetector(
        model_path=model_path,
        device=device,
        buffer_size=settings.buffer_size,
        threshold=settings.min_lstm_probability
    )
    
    return detector


if __name__ == "__main__":
    # Test inference loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fall detection inference")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load detector
    detector = RealTimeFallDetector(
        model_path=args.model_path,
        device=args.device
    )
    
    print("="*50)
    print("Detector State:")
    print("="*50)
    state = detector.get_state()
    for key, value in state.items():
        print(f"{key}: {value}")
    print("="*50)
