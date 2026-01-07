"""
Training Pipeline for Fall Detection LSTM

Handles:
- Data loading
- Model training with early stopping
- Class balancing
- Learning rate scheduling
- Model evaluation
- Checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from .lstm_model import FallDetectionLSTM, EarlyStopping
from ..utils.config import TRAINING_CONFIG, settings
from ..utils.logger import setup_logger, LoggerMixin


logger = setup_logger(__name__)


class FallDataset(Dataset):
    """PyTorch dataset for fall detection."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Features of shape (N, time_steps, features)
            y: Labels of shape (N,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class FallDetectionTrainer(LoggerMixin):
    """Trainer for fall detection LSTM model."""
    
    def __init__(
        self,
        model: FallDetectionLSTM,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: LSTM model to train
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Loss function (with optional class weights)
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=TRAINING_CONFIG["reduce_lr_factor"],
            patience=TRAINING_CONFIG["reduce_lr_patience"],
            min_lr=TRAINING_CONFIG["min_lr"],
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG["early_stopping_patience"],
            mode='min'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        self.logger.info(f"Trainer initialized on {self.device}")
    
    def create_weighted_sampler(self, y: np.ndarray) -> WeightedRandomSampler:
        """
        Create weighted sampler for class balancing.
        
        Args:
            y: Labels array
        
        Returns:
            WeightedRandomSampler
        """
        # Count classes
        unique, counts = np.unique(y, return_counts=True)
        
        # Compute weights (inverse frequency)
        class_weights = 1.0 / counts
        
        # Apply additional weighting for falls
        if TRAINING_CONFIG["use_class_weights"]:
            fall_idx = np.where(unique == 1)[0]
            if len(fall_idx) > 0:
                class_weights[fall_idx[0]] *= TRAINING_CONFIG["fall_weight"]
        
        # Sample weights
        sample_weights = np.array([class_weights[int(label)] for label in y])
        
        self.logger.info(f"Class weights: {dict(zip(unique, class_weights))}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Compute metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0)
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        save_dir: Path = Path("./trained_models")
    ) -> Dict:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save model
        
        Returns:
            Training history
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
                f"Val Recall: {val_metrics['recall']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                
                checkpoint_path = save_dir / "best_model.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                self.logger.info(f"✓ New best model saved (Val Loss: {val_metrics['loss']:.4f})")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model
        final_path = save_dir / "lstm_fall_detector.pth"
        self.save_checkpoint(final_path, epoch, val_metrics, is_final=True)
        
        # Save history
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info("Training complete!")
        
        return self.history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict,
        is_final: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'model_config': self.model.get_model_info()
        }
        
        torch.save(checkpoint, path)
    
    def evaluate_test_set(
        self,
        test_loader: DataLoader
    ) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Returns:
            Dictionary with all metrics and confusion matrix
        """
        self.logger.info("Evaluating on test set...")
        
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                preds = (outputs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Flatten arrays
        all_preds = np.array(all_preds).flatten()
        all_probs = np.array(all_probs).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)
        
        # Classification report
        report = classification_report(
            all_targets, all_preds,
            target_names=['Non-Fall', 'Fall'],
            output_dict=True
        )
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Log results
        self.logger.info("="*60)
        self.logger.info("TEST SET RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        self.logger.info(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        self.logger.info(f"F1 Score:  {f1:.4f}")
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        self.logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        self.logger.info("="*60)
        
        # Check if targets are met
        if precision >= TRAINING_CONFIG["target_precision"]:
            self.logger.info(f"✓ Precision target met: {precision:.4f} >= {TRAINING_CONFIG['target_precision']}")
        else:
            self.logger.warning(f"✗ Precision below target: {precision:.4f} < {TRAINING_CONFIG['target_precision']}")
        
        if recall >= TRAINING_CONFIG["target_recall"]:
            self.logger.info(f"✓ Recall target met: {recall:.4f} >= {TRAINING_CONFIG['target_recall']}")
        else:
            self.logger.warning(f"✗ Recall below target: {recall:.4f} < {TRAINING_CONFIG['target_recall']}")
        
        return results


def load_dataset(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load prepared dataset."""
    logger.info(f"Loading dataset from {data_dir}")
    
    dataset = {}
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / f"X_{split}.npy")
        y = np.load(data_dir / f"y_{split}.npy")
        dataset[split] = (X, y)
        logger.info(f"{split}: {len(X)} samples")
    
    return dataset


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fall detection LSTM")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./trained_models"))
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG["learning_rate"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bidirectional", action="store_true")
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset(args.data_dir)
    X_train, y_train = dataset['train']
    X_val, y_val = dataset['val']
    X_test, y_test = dataset['test']
    
    # Create datasets
    train_dataset = FallDataset(X_train, y_train)
    val_dataset = FallDataset(X_val, y_val)
    test_dataset = FallDataset(X_test, y_test)
    
    # Create model
    from .lstm_model import create_model
    model = create_model(
        input_dim=7,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3,
        bidirectional=args.bidirectional,
        device=device
    )
    
    logger.info(f"Model: {model.get_model_info()}")
    
    # Create trainer
    trainer = FallDetectionTrainer(
        model=model,
        device=device,
        learning_rate=args.lr
    )
    
    # Create weighted sampler for training
    sampler = trainer.create_weighted_sampler(y_train)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.output_dir
    )
    
    # Evaluate on test set
    test_results = trainer.evaluate_test_set(test_loader)
    
    # Save test results
    results_path = args.output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test results saved to {results_path}")


if __name__ == "__main__":
    main()
