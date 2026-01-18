# Armenian Handwriting Recognition using ResNet

A deep learning project for classifying Armenian handwritten characters using a ResNet-like architecture trained on the Mashtots dataset.

## Project Overview

This notebook implements:

- **Dataset**: Mashtots Armenian handwriting dataset (78 character classes)
- **Model**: Custom ResNet architecture with residual blocks and group normalization
- **Task**: Train a classifier and make predictions on test data
- **Output**: Submission files with predictions and confidence scores

## Project Structure

```
Armenian_handwrite.ipynb          # Main notebook with complete pipeline
mashtots-dataset/                 # Dataset directory
├── Train/                        # Training data (80% of original)
├── Test/                         # Validation data (20% split from Train)
└── new_test/                     # Test set for predictions
mashtots_resnet.pth              # Saved trained model weights
submission.csv                    # Final predictions
detailed_predictions.csv          # Predictions with confidence scores
```

## Setup Instructions

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, CPU supported)
- Kaggle API credentials

### Installation

1. **Install Dependencies**

   ```bash
   pip install torch torchvision
   pip install pandas numpy scikit-learn
   pip install opencv-python albumentations
   pip install kaggle matplotlib
   ```
2. **Configure Kaggle API**

   - Download `kaggle.json` from your Kaggle account settings
   - Place it in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. **Download Dataset**

   ```bash
   kaggle competitions download -c mashtots-dataset
   unzip mashtots-dataset.zip
   ```

## Configuration

Key hyperparameters (configurable in the notebook):

| Parameter       | Value   | Description                     |
| --------------- | ------- | ------------------------------- |
| `IMG_SIZE`    | 64      | Image resolution (64x64 pixels) |
| `NUM_CLASSES` | 78      | Armenian character classes      |
| `BATCH_SIZE`  | 64      | Training batch size             |
| `EPOCHS`      | 30      | Training epochs                 |
| `LR`          | 1e-3    | Learning rate                   |
| `DEVICE`      | GPU/CPU | Auto-detected                   |

## Model Architecture

### ResidualBlock

- 3x3 convolutions with group normalization
- Skip connections for gradient flow
- Configurable stride and channel expansion

### MashtotsResNet

```
Input (1, 64, 64)
    ↓
Stem: Conv2d(1→64) + GroupNorm + ReLU
    ↓
ResidualBlock(64→64)
    ↓
ResidualBlock(64→128, stride=2)
    ↓
ResidualBlock(128→256, stride=2)
    ↓
ResidualBlock(256→512, stride=2)
    ↓
AdaptiveAvgPool2d → FC(512→78)
    ↓
Output: 78 class logits
```

## Training Pipeline

1. **Data Preparation**

   - Split training data: 80% train, 20% validation
   - Augmentation for training: rotation, shift, blur, brightness/contrast
2. **Training Loop**

   - Optimizer: AdamW with weight decay (1e-4)
   - Scheduler: Cosine annealing LR
   - Loss: CrossEntropyLoss
   - Metrics: Accuracy per epoch
3. **Model Checkpoint**

   - Best model saved to `mashtots_resnet.pth`

## Testing & Predictions

1. **Test Time Augmentation (TTA)**

   - Original image
   - Horizontally flipped image
   - Average predictions for robustness
2. **Outputs**

   - `submission.csv`: Predicted class labels
   - `detailed_predictions.csv`: Predictions + confidence scores

## Usage

Simply run the notebook cells in order:

```
1. Download & setup Kaggle credentials
2. Download & extract dataset
3. Import libraries & set config
4. Split train/test data
5. Define model architecture
6. Prepare DataLoaders
7. Train model
8. Save model weights
9. Load model & make predictions
10. Generate submission files
```

## Results Metrics

- **Mean Confidence**: Overall model confidence on test set
- **Correct Prediction Confidence**: Confidence on correct predictions
- **Wrong Prediction Confidence**: Confidence on incorrect predictions

## Notes

- Images are read as grayscale (1 channel)
- Normalization: mean=0.5, std=0.5
- Group normalization (8 groups) used instead of batch norm for stability
- GPU acceleration recommended for faster training

## Author

Armenian Handwriting Recognition Project

## License

Dataset: Mashtots Dataset
