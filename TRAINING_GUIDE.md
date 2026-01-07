# 🚀 Complete Training and Deployment Guide

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB minimum
- **Storage**: 50GB+ free space

### Software Requirements
- Python 3.10+
- Node.js 18+
- CUDA 11.8+ (for GPU support)
- Git

## 🗂️ Step 1: Download Datasets

Download these three datasets and place them in the `datasets/` folder:

### 1. UR Fall Detection Dataset
- **Link**: [UR Fall Detection](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html)
- Extract to: `datasets/UR_Fall/`

### 2. UP-Fall Detection Dataset  
- **Link**: [UP-Fall Dataset](https://sites.google.com/up.edu.mx/har-up/)
- Extract to: `datasets/UP_Fall/`

### 3. Le2i Fall Detection Dataset
- **Link**: [Le2i Dataset](http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr)
- Extract to: `datasets/Le2i/`

**Expected structure:**
```
datasets/
├── UR_Fall/
│   ├── fall/
│   └── adl/
├── UP_Fall/
│   ├── Falls/
│   └── ADLs/
└── Le2i/
    ├── fall/
    └── not_fall/
```

## 🔧 Step 2: Environment Setup

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp ../.env.example .env

# Edit .env file with your settings
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

## 📊 Step 3: Process Datasets

### 3.1 Extract Frames from Videos

```bash
cd backend

python -m data_processing.dataset_loader \
  --dataset-root ../datasets \
  --output-root ./data/processed \
  --target-fps 10
```

**Expected output:**
- Processed frames metadata in `data/processed/metadata/`
- Processing statistics in `data/processed/processing_stats.json`

### 3.2 Extract Poses with YOLO

```bash
python -m data_processing.pose_extractor \
  --metadata-dir ./data/processed/metadata \
  --output-dir ./data/processed/poses \
  --dataset-root ../datasets \
  --device cuda
```

**Time estimate:** 2-4 hours depending on dataset size and GPU

**Expected output:**
- Pose data in `data/processed/poses/*_poses.json`
- Statistics in `data/processed/poses/pose_extraction_stats.json`

### 3.3 Extract Features

```bash
python -m data_processing.feature_engineer \
  --poses-dir ./data/processed/poses \
  --output-dir ./data/processed/features
```

**Expected output:**
- Feature vectors in `data/processed/features/*_features.json`

### 3.4 Create Temporal Dataset

```bash
python -m data_processing.temporal_dataset \
  --features-dir ./data/processed/features \
  --metadata-dir ./data/processed/metadata \
  --output-dir ./data/dataset \
  --window-size 30 \
  --stride 5 \
  --augment
```

**Expected output:**
- Training data: `data/dataset/X_train.npy`, `y_train.npy`
- Validation data: `data/dataset/X_val.npy`, `y_val.npy`
- Test data: `data/dataset/X_test.npy`, `y_test.npy`
- Statistics: `data/dataset/dataset_metadata.json`

## 🧠 Step 4: Train LSTM Model

```bash
python -m models.train \
  --data-dir ./data/dataset \
  --output-dir ./trained_models \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --device cuda
```

**Training progress:**
- Model checkpoints saved every epoch
- Best model: `trained_models/best_model.pth`
- Final model: `trained_models/lstm_fall_detector.pth`
- Training history: `trained_models/training_history.json`
- Test results: `trained_models/test_results.json`

**Expected metrics:**
- Precision: ≥97%
- Recall: ≥95%
- F1 Score: ≥96%

⚠️ **If metrics don't meet targets**, retrain with:
- More augmentation
- Different learning rate
- Longer training (more epochs)

## 🚀 Step 5: Start Services

### Option A: Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

python -m api.main
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option B: Production Mode (Docker)

```bash
# Build and start services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Step 6: Testing

### Test Backend API

```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/session/create

# Get stats
curl http://localhost:8000/stats
```

### Test Frontend

1. Open http://localhost:5173
2. Click "Start Monitoring"
3. Allow camera access
4. Monitor detection status in sidebar

### Test Fall Detection

1. Start monitoring
2. Simulate fall in front of camera:
   - Lean forward
   - Lower to ground
   - Stay still for 5+ seconds
3. Alert should trigger after 2 seconds of high confidence

## 📈 Step 7: Monitoring & Optimization

### View Training Metrics

```python
import json

# Load training history
with open('trained_models/training_history.json', 'r') as f:
    history = json.load(f)

# Load test results
with open('trained_models/test_results.json', 'r') as f:
    results = json.load(f)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test Precision: {results['precision']:.4f}")
print(f"Test Recall: {results['recall']:.4f}")
print(f"Test F1: {results['f1_score']:.4f}")
```

### Adjust Detection Sensitivity

Edit `.env` file:

```env
# More sensitive (more detections, possibly more false positives)
FALL_THRESHOLD=0.80
MIN_LSTM_PROBABILITY=0.75

# Less sensitive (fewer false positives, might miss some falls)
FALL_THRESHOLD=0.90
MIN_LSTM_PROBABILITY=0.90
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, set device to CPU in .env
DEVICE=cpu
```

### Model Not Loading

```bash
# Check model file exists
ls trained_models/lstm_fall_detector.pth

# Check path in .env
MODEL_PATH=./trained_models/lstm_fall_detector.pth
```

### WebSocket Connection Fails

```bash
# Check backend is running
curl http://localhost:8000/health

# Check CORS settings in .env
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Low Accuracy

1. **Collect more data**: Add more diverse fall scenarios
2. **Increase augmentation**: Edit `TRAINING_CONFIG` in `config.py`
3. **Adjust class weights**: Increase `fall_weight` for rare fall samples
4. **Train longer**: Increase epochs
5. **Fine-tune thresholds**: Adjust validation parameters

## 📊 Performance Benchmarks

### Expected Performance (RTX 3060)

- **Training**: ~2-3 hours for 100 epochs
- **Inference**: ~15-20ms per frame
- **Total Latency**: <200ms (frame capture → result)
- **Throughput**: 8-10 FPS

### Memory Usage

- **Training**: 4-6GB GPU VRAM
- **Inference**: 2-3GB GPU VRAM
- **CPU Mode**: 8GB RAM minimum

## 🔐 Security Considerations

### Before Production Deployment

1. **Change SECRET_KEY** in `.env`
2. **Enable HTTPS** (use reverse proxy like Nginx)
3. **Set proper CORS_ORIGINS**
4. **Enable authentication** if multi-user
5. **Configure alert channels** (email, SMS)
6. **Set up monitoring** (Prometheus, Grafana)
7. **Implement rate limiting**
8. **Add request logging**

### Privacy Compliance

- ✅ No video recording (frames discarded after processing)
- ✅ No image storage
- ✅ Only pose data processed
- ✅ Configurable data retention
- ✅ GDPR compliant

## 📞 Support

### Check Logs

```bash
# Backend logs
tail -f logs/fall_detection.log

# Docker logs
docker-compose logs -f backend
```

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size or use CPU |
| Low FPS | Check GPU utilization, reduce frame size |
| False positives | Increase thresholds, adjust physics rules |
| Missed falls | Lower thresholds, check lighting conditions |
| Camera not working | Check browser permissions |

## 🎉 Success Criteria

Your system is production-ready when:

- ✅ Test precision ≥97%
- ✅ Test recall ≥95%
- ✅ Inference latency <200ms
- ✅ WebSocket connection stable
- ✅ Alert system responsive
- ✅ False alarms <1 per day in testing

## 📝 Next Steps

1. **Test extensively** with real users
2. **Collect feedback** on false positives/negatives
3. **Fine-tune thresholds** based on environment
4. **Set up monitoring** infrastructure
5. **Configure alert channels**
6. **Document deployment** for your team
7. **Plan maintenance** schedule

---

**🎓 You now have a complete production-grade fall detection system!**
