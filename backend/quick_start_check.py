"""
Quick Start Script

Run this to verify your setup and test the basic functionality.
"""

import sys
import subprocess
from pathlib import Path
import importlib


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False


def check_dependencies():
    """Check if all dependencies are installed."""
    print("\nChecking dependencies...")
    
    required = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('ultralytics', 'YOLOv8'),
        ('fastapi', 'FastAPI'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn')
    ]
    
    missing = []
    for module, name in required:
        try:
            importlib.import_module(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("⚠ CUDA not available, will use CPU (slower)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_datasets():
    """Check if datasets are present."""
    print("\nChecking datasets...")
    
    datasets_dir = Path("../datasets")
    required_datasets = ['UR_Fall', 'UP_Fall', 'Le2i']
    
    found = []
    missing = []
    
    for dataset in required_datasets:
        dataset_path = datasets_dir / dataset
        if dataset_path.exists():
            print(f"✓ {dataset}")
            found.append(dataset)
        else:
            print(f"✗ {dataset}")
            missing.append(dataset)
    
    if missing:
        print(f"\n⚠ Missing datasets: {', '.join(missing)}")
        print("See TRAINING_GUIDE.md for download instructions")
        return False
    
    return True


def check_model():
    """Check if trained model exists."""
    print("\nChecking trained model...")
    
    model_path = Path("./trained_models/lstm_fall_detector.pth")
    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1e6:.1f}MB")
        return True
    else:
        print(f"✗ Model not found: {model_path}")
        print("You need to train the model first")
        print("See TRAINING_GUIDE.md for instructions")
        return False


def check_directories():
    """Check required directories."""
    print("\nChecking directories...")
    
    required_dirs = [
        'data/processed',
        'trained_models',
        'logs'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")
    
    return True


def test_yolo():
    """Test YOLOv8 installation."""
    print("\nTesting YOLOv8...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8m-pose.pt')
        print("✓ YOLOv8 working")
        return True
    except Exception as e:
        print(f"✗ YOLOv8 error: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("Fall Detection System - Quick Start Check")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA Support", check_cuda),
        ("Datasets", check_datasets),
        ("Trained Model", check_model),
        ("Directories", check_directories),
        ("YOLOv8", test_yolo)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("1. If model not trained: Follow TRAINING_GUIDE.md")
        print("2. Start backend: python -m api.main")
        print("3. Start frontend: cd ../frontend && npm run dev")
        print("4. Open: http://localhost:5173")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        if not results.get("Trained Model", False):
            print("\n📚 To train the model:")
            print("   See TRAINING_GUIDE.md for step-by-step instructions")
    
    print("="*60)


if __name__ == "__main__":
    main()
