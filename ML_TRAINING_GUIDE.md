# Smart Presence ML Model Training Guide

This guide explains how to train the face detection and emotion recognition models for the Smart Presence system using the provided datasets.

## 📁 Dataset Structure

### Face Detection Dataset
```
datasets/
├── face detection/
│   ├── images/
│   │   ├── train/        # Training images (.jpg)
│   │   └── val/          # Validation images (.jpg)
│   └── labels/
│       ├── train/        # Training labels (.txt, YOLO format)
│       └── val/          # Validation labels (.txt, YOLO format)
```

### Emotion Detection Dataset
```
datasets/
├── Emotion/
│   ├── train/
│   │   ├── angry/        # Angry emotion images (.png)
│   │   ├── disgusted/    # Disgusted emotion images (.png)
│   │   ├── fearful/      # Fearful emotion images (.png)
│   │   ├── happy/        # Happy emotion images (.png)
│   │   ├── neutral/      # Neutral emotion images (.png)
│   │   ├── sad/          # Sad emotion images (.png)
│   │   └── surprised/    # Surprised emotion images (.png)
│   └── test/             # Test data (optional)
```

## 🚀 Quick Start

### 1. Check System Requirements

First, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset Availability

Before training, check if your datasets are properly organized:

```bash
python manage.py train_models --check-only
```

This will display:
- ✅ Available datasets and models
- ❌ Missing components
- 📊 Dataset statistics

### 3. Train Models

#### Option A: Train Both Models (Recommended)
```bash
python manage.py train_models --all --epochs 100
```

#### Option B: Train Individual Models
```bash
# Face detection only
python manage.py train_models --face --epochs 100

# Emotion recognition only
python manage.py train_models --emotion --epochs 150
```

#### Option C: Custom Training Parameters
```bash
python manage.py train_models --all --epochs 100 --batch-size 32 --learning-rate 0.001
```

## 🔧 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Batch size for training |
| `--learning-rate` | 0.001 | Learning rate for optimization |

### Recommended Settings

**For Face Detection:**
- Epochs: 100-200
- Batch Size: 16-32 (depending on GPU memory)
- Learning Rate: 0.001

**For Emotion Recognition:**
- Epochs: 150-300
- Batch Size: 32-64
- Learning Rate: 0.001

## 🏗️ Model Architectures

### Face Detection Model
- **Type**: CNN-based bounding box regression
- **Input**: 224x224 RGB images
- **Output**: Bounding box coordinates + confidence score
- **Loss**: Combined Huber loss (bbox) + Binary cross-entropy (confidence)

**Architecture:**
```
Conv2D(32) → BatchNorm → MaxPool2D
Conv2D(64) → BatchNorm → MaxPool2D
Conv2D(128) → BatchNorm → MaxPool2D
Conv2D(256) → BatchNorm → MaxPool2D
Conv2D(512) → BatchNorm → MaxPool2D
GlobalAveragePooling2D
Dense(512) → Dropout(0.5)
Dense(256) → Dropout(0.3)
Dense(5)  # 4 bbox coords + 1 confidence
```

### Emotion Recognition Model
- **Type**: CNN classifier
- **Input**: 48x48 RGB images (grayscale converted)
- **Output**: 7-class emotion probabilities
- **Loss**: Categorical cross-entropy

**Architecture:**
```
Data Augmentation Layer
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool2D → Dropout(0.25)
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool2D → Dropout(0.25)
Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm → MaxPool2D → Dropout(0.25)
Conv2D(512) → BatchNorm → Conv2D(512) → BatchNorm → MaxPool2D → Dropout(0.25)
GlobalAveragePooling2D
Dense(512) → BatchNorm → Dropout(0.5)
Dense(256) → BatchNorm → Dropout(0.5)
Dense(128) → Dropout(0.3)
Dense(7)  # 7 emotion classes
```

## 📊 Training Output

During training, you'll see:

1. **Dataset Statistics**: Number of samples loaded
2. **Model Architecture**: Layer-by-layer summary
3. **Training Progress**: Epoch-by-epoch loss and accuracy
4. **Validation Metrics**: Real-time validation performance
5. **Training Plots**: Saved to `models/` directory
6. **Model Checkpoints**: Best models saved automatically

### Generated Files
```
models/
├── face_detection_model.h5              # Trained face detection model
├── face_detection_best.h5               # Best checkpoint
├── face_detection_training_history.png  # Training plots
├── emotion_detection_model.h5           # Trained emotion model
├── emotion_detection_best.h5            # Best checkpoint
├── emotion_detection_training_history.png # Training plots
├── emotion_detection_confusion_matrix.png # Confusion matrix
├── emotion_classification_report.json   # Detailed metrics
└── model_config.json                    # Model configuration
```

## ⚡ Performance Optimization

### GPU Training
Ensure TensorFlow can access your GPU:

```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

### Memory Optimization
- Reduce batch size if encountering memory errors
- Use mixed precision training for faster training:

```bash
export TF_ENABLE_AUTO_MIXED_PRECISION=1
```

### Training Tips

1. **Start Small**: Begin with fewer epochs to test the pipeline
2. **Monitor Overfitting**: Watch validation loss vs training loss
3. **Data Augmentation**: Enabled by default for emotion recognition
4. **Early Stopping**: Automatically stops if no improvement
5. **Learning Rate Scheduling**: Automatically reduces LR when stuck

## 🎯 Expected Performance

### Face Detection
- **Training Time**: 2-4 hours (100 epochs, GPU)
- **Expected Accuracy**: 85-95% detection rate
- **Inference Speed**: ~50-100 FPS (GPU), ~10-20 FPS (CPU)

### Emotion Recognition
- **Training Time**: 1-3 hours (150 epochs, GPU)
- **Expected Accuracy**: 70-85% (7-class classification)
- **Inference Speed**: ~100-200 FPS (GPU), ~20-50 FPS (CPU)

## 🔍 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   Error: Face detection dataset not found
   ```
   - Verify dataset folder structure
   - Check file permissions
   - Ensure correct naming conventions

2. **Memory Errors**
   ```
   OOM when allocating tensor
   ```
   - Reduce batch size: `--batch-size 16`
   - Close other applications
   - Use smaller input sizes

3. **Training Stuck**
   ```
   Loss not decreasing
   ```
   - Check learning rate (try 0.0001)
   - Verify data quality
   - Increase epochs

4. **Model Loading Errors**
   ```
   Model file not found
   ```
   - Ensure training completed successfully
   - Check models/ directory permissions
   - Retrain if files are corrupted

## 🧪 Testing Trained Models

After training, test your models:

```python
from ml_models.inference import get_inference_instance

# Initialize inference
inference = get_inference_instance()

# Test with an image
result = inference.process_image('path/to/test/image.jpg')
print(f"Face detected: {result['face_detected']}")
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['emotion_confidence']}")
```

## 📈 Model Evaluation

### Metrics Tracked

**Face Detection:**
- Validation Loss
- Mean Absolute Error (MAE)
- Detection Confidence

**Emotion Recognition:**
- Accuracy
- Top-3 Accuracy
- Per-class Precision/Recall
- Confusion Matrix

### Evaluation Scripts

Run evaluation on test data:

```bash
python evaluate_models.py --model-type face
python evaluate_models.py --model-type emotion
python evaluate_models.py --model-type both
```

## 🔄 Retraining

To retrain models with new data:

1. Add new data to appropriate directories
2. Run training again (will overwrite existing models)
3. Models automatically save best checkpoints

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs in `training.log`
3. Verify dataset integrity
4. Check system requirements

---

**Happy Training! 🚀**

The trained models will be automatically integrated into the Smart Presence web application for real-time face detection and emotion recognition during attendance scanning. 