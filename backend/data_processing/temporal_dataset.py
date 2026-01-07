"""
Temporal Dataset Creation

Creates sliding window sequences for LSTM training.
Converts feature sequences into (X, y) pairs.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

from ..utils.config import settings, TRAINING_CONFIG
from ..utils.logger import setup_logger, LoggerMixin


logger = setup_logger(__name__)


class TemporalDataset(LoggerMixin):
    """
    Create temporal datasets for LSTM training.
    
    Uses sliding window approach:
    - Window size: 30-45 frames
    - Stride: 5 frames
    - Label: 0 (non-fall) or 1 (fall)
    """
    
    def __init__(
        self,
        window_size: int = 30,
        stride: int = 5,
        feature_dim: int = 7
    ):
        """
        Initialize temporal dataset creator.
        
        Args:
            window_size: Number of frames in each window
            stride: Step size for sliding window
            feature_dim: Dimension of feature vectors
        """
        self.window_size = window_size
        self.stride = stride
        self.feature_dim = feature_dim
        
        self.logger.info(f"Temporal dataset: window={window_size}, stride={stride}")
    
    def load_features_from_file(self, feature_file: Path) -> Tuple[np.ndarray, List[float]]:
        """
        Load feature vectors from JSON file.
        
        Returns:
            Tuple of (features array, timestamps)
        """
        with open(feature_file, 'r') as f:
            data = json.load(f)
        
        features = []
        timestamps = []
        
        for feat in data["features"]:
            features.append(feat["vector"])
            timestamps.append(feat["timestamp"])
        
        return np.array(features, dtype=np.float32), timestamps
    
    def create_windows(
        self,
        features: np.ndarray,
        label: int,
        min_window_size: int = None
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Create sliding windows from feature sequence.
        
        Args:
            features: Feature array of shape (T, feature_dim)
            label: Class label (0 or 1)
            min_window_size: Minimum window size to accept
        
        Returns:
            List of (window, label) tuples
        """
        if min_window_size is None:
            min_window_size = self.window_size
        
        if len(features) < min_window_size:
            self.logger.warning(f"Sequence too short: {len(features)} < {min_window_size}")
            return []
        
        windows = []
        
        # Slide window through sequence
        for i in range(0, len(features) - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            windows.append((window, label))
        
        return windows
    
    def load_and_create_dataset(
        self,
        features_dir: Path,
        metadata_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all features and create dataset.
        
        Args:
            features_dir: Directory with feature JSON files
            metadata_dir: Directory with video metadata (for labels)
        
        Returns:
            Tuple of (X, y) where:
                X: shape (N, window_size, feature_dim)
                y: shape (N,) with labels 0 or 1
        """
        feature_files = list(features_dir.glob("*_features.json"))
        if not feature_files:
            raise ValueError(f"No feature files found in {features_dir}")
        
        self.logger.info(f"Loading {len(feature_files)} feature files")
        
        all_windows = []
        all_labels = []
        
        fall_count = 0
        non_fall_count = 0
        
        for feature_file in tqdm(feature_files, desc="Creating windows"):
            try:
                # Load features
                features, _ = self.load_features_from_file(feature_file)
                
                # Get label from metadata
                metadata_file = metadata_dir / feature_file.name.replace("_features.json", ".json")
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                label = 1 if metadata["is_fall"] else 0
                
                # Create windows
                windows = self.create_windows(features, label)
                
                for window, lbl in windows:
                    all_windows.append(window)
                    all_labels.append(lbl)
                    
                    if lbl == 1:
                        fall_count += 1
                    else:
                        non_fall_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process {feature_file}: {e}")
        
        X = np.array(all_windows, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        
        self.logger.info(f"Dataset created: {len(X)} windows")
        self.logger.info(f"Falls: {fall_count}, Non-falls: {non_fall_count}")
        self.logger.info(f"Class imbalance ratio: 1:{non_fall_count/max(fall_count, 1):.1f}")
        
        return X, y
    
    def apply_augmentation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment_config: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to training data.
        
        Augmentation strategies:
        1. Temporal jitter: Randomly skip frames
        2. Speed variation: Interpolate to change speed
        3. Pose noise: Add small Gaussian noise
        4. Horizontal flip: Mirror left/right
        
        Args:
            X: Input features (N, T, F)
            y: Labels (N,)
            augment_config: Augmentation configuration
        
        Returns:
            Augmented (X, y)
        """
        augmented_X = []
        augmented_y = []
        
        # Keep original data
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Augment fall samples more (they're rare)
        fall_mask = y == 1
        fall_X = X[fall_mask]
        fall_y = y[fall_mask]
        
        self.logger.info(f"Augmenting {len(fall_X)} fall samples")
        
        # 1. Pose noise augmentation
        if augment_config.get("pose_noise", 0) > 0:
            noise_std = augment_config["pose_noise"]
            noisy_X = fall_X + np.random.randn(*fall_X.shape) * noise_std
            augmented_X.append(noisy_X)
            augmented_y.append(fall_y)
        
        # 2. Speed variation (simple subsampling)
        if "speed_variation" in augment_config:
            min_speed, max_speed = augment_config["speed_variation"]
            
            # Slower version (0.8x speed)
            slow_indices = np.linspace(0, self.window_size - 1, int(self.window_size * min_speed)).astype(int)
            slow_X = fall_X[:, slow_indices, :]
            # Pad to original length
            if len(slow_indices) < self.window_size:
                pad_width = ((0, 0), (0, self.window_size - len(slow_indices)), (0, 0))
                slow_X = np.pad(slow_X, pad_width, mode='edge')
            augmented_X.append(slow_X)
            augmented_y.append(fall_y)
        
        # Concatenate all augmentations
        X_aug = np.concatenate(augmented_X, axis=0)
        y_aug = np.concatenate(augmented_y, axis=0)
        
        self.logger.info(f"After augmentation: {len(X_aug)} samples")
        
        return X_aug, y_aug
    
    def normalize_features(
        self,
        X: np.ndarray,
        mean: np.ndarray = None,
        std: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features to zero mean and unit variance.
        
        Args:
            X: Input features (N, T, F)
            mean: Pre-computed mean (for test set)
            std: Pre-computed std (for test set)
        
        Returns:
            Normalized X, mean, std
        """
        if mean is None or std is None:
            # Compute statistics from training data
            mean = X.mean(axis=(0, 1), keepdims=True)
            std = X.std(axis=(0, 1), keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero
        
        X_normalized = (X - mean) / std
        
        return X_normalized, mean, std
    
    def create_train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            X: Features (N, T, F)
            y: Labels (N,)
            val_split: Validation set ratio
            test_split: Test set ratio
            random_seed: Random seed for reproducibility
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_split,
            stratify=y,
            random_state=random_seed
        )
        
        # Second split: train vs val
        val_size = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=random_seed
        )
        
        self.logger.info(f"Train: {len(X_train)} samples")
        self.logger.info(f"Val: {len(X_val)} samples")
        self.logger.info(f"Test: {len(X_test)} samples")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def save_dataset(
        self,
        dataset: Dict,
        output_dir: Path,
        normalization_stats: Dict = None
    ):
        """Save processed dataset to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        for split_name, (X, y) in dataset.items():
            np.save(output_dir / f"X_{split_name}.npy", X)
            np.save(output_dir / f"y_{split_name}.npy", y)
        
        # Save normalization statistics
        if normalization_stats:
            with open(output_dir / "normalization_stats.pkl", 'wb') as f:
                pickle.dump(normalization_stats, f)
        
        # Save metadata
        metadata = {
            "window_size": self.window_size,
            "stride": self.stride,
            "feature_dim": self.feature_dim,
            "splits": {
                split_name: {
                    "samples": len(X),
                    "falls": int(np.sum(y)),
                    "non_falls": int(np.sum(y == 0))
                }
                for split_name, (X, y) in dataset.items()
            }
        }
        
        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Dataset saved to {output_dir}")


def main():
    """Main function to create temporal dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create temporal dataset for LSTM training")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--augment", action="store_true")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset_creator = TemporalDataset(
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Load and create windows
    X, y = dataset_creator.load_and_create_dataset(
        features_dir=args.features_dir,
        metadata_dir=args.metadata_dir
    )
    
    # Apply augmentation if requested
    if args.augment:
        X, y = dataset_creator.apply_augmentation(
            X, y, TRAINING_CONFIG["augmentation"]
        )
    
    # Normalize
    X, mean, std = dataset_creator.normalize_features(X)
    
    # Split dataset
    splits = dataset_creator.create_train_val_test_split(
        X, y,
        val_split=TRAINING_CONFIG["validation_split"],
        test_split=TRAINING_CONFIG["test_split"],
        random_seed=TRAINING_CONFIG["random_seed"]
    )
    
    # Save
    dataset_creator.save_dataset(
        splits,
        args.output_dir,
        normalization_stats={"mean": mean, "std": std}
    )
    
    print("\n" + "="*50)
    print("DATASET CREATION COMPLETE")
    print("="*50)
    for split_name, (X_split, y_split) in splits.items():
        print(f"{split_name.upper()}: {len(X_split)} samples")
        print(f"  Falls: {np.sum(y_split)}")
        print(f"  Non-falls: {np.sum(y_split == 0)}")
    print("="*50)


if __name__ == "__main__":
    main()
