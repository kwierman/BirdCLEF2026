# BirdCLEF2026 Notebooks

This directory contains comprehensive Jupyter notebooks for the BirdCLEF2026 Challenge - Wildlife Species Classification from Audio.

## Challenge Overview

- **Goal**: Identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from audio recordings
- **Data**: 1-minute ogg audio files at 32kHz from Pantanal wetlands, Brazil
- **Evaluation**: Predictions for 5-second segments in test soundscapes

## Notebooks

| Notebook | Description | Key Decisions |
|----------|-------------|----------------|
| `01_research.ipynb` | Research findings, external resources, model selection rationale | Based on BirdCLEF 2024/2025 winning approaches |
| `02_eda.ipynb` | Exploratory data analysis - class distribution, taxonomy, data quality | Identified domain shift challenge |
| `03_data_preprocessing.ipynb` | Audio loading, mel spectrograms, augmentation pipeline | Chose mel spectrograms as standard bioacoustic features |
| `04_model_training.ipynb` | EfficientNet-B0 training with validation | Combined train_audio + train_soundscapes for domain adaptation |
| `05_evaluation_submission.ipynb` | Model inference and submission generation | Handles missing test files gracefully |

## Key Insights from Research & EDA

### 1. Domain Shift (Critical)
- Training data: Clean Xeno-canto/iNaturalist recordings
- Test data: Noisy field recordings from Pantanal
- Solution: Combine both data sources for training

### 2. Class Imbalance
- Many species have <10 samples
- Need data augmentation and weighted loss

### 3. Multi-label
- Field recordings have overlapping species
- Use BCE loss (not softmax)

### 4. Species Only in Soundscapes
- Some test species only appear in train_soundscapes
- Must use this data for training!

## Approach Summary

### Feature Extraction
- Mel spectrograms: 128 mel bands, 5-second windows, 32kHz
- Standard in bioacoustic classification

### Model
- EfficientNet-B0 (pretrained on ImageNet)
- ~5M parameters, fast training

### Training
- Combined dataset (train_audio + train_soundscapes)
- Data augmentation: noise, pitch shift, time stretch
- BCE loss for multi-label
- AdamW optimizer with cosine annealing
- Validation AUC metric

### Inference
- 12 segments per 1-minute test file
- Predict probability for each of 234 species

## How to Run

1. **01_research.ipynb** - Run first to understand the approach
2. **02_eda.ipynb** - Explore data characteristics
3. **03_data_preprocessing.ipynb** - Understand preprocessing
4. **04_model_training.ipynb** - Train model (requires GPU)
5. **05_evaluation_submission.ipynb** - Generate submission.csv

## Requirements

See `requirements.txt` for Python dependencies:
- torch, timm, transformers
- librosa, soundfile, audiomentations
- pandas, numpy, scikit-learn

## Data Structure

```
../
├── train_audio/              # Training audio files (Xeno-canto, iNaturalist)
├── train_soundscapes/       # Field recordings with labels
├── test_soundscapes/         # Test recordings (populated at submission time)
├── train.csv                 # Training metadata
├── taxonomy.csv              # Species taxonomy
├── train_soundscapes_labels.csv  # Labeled soundscape segments
├── sample_submission.csv     # Submission template
└── submission.csv            # Final submission (generated)
```

## Output

- `output/models/efficientnet_best.pt` - Trained model weights
- `output/training_config.json` - Training configuration
- `submission.csv` - Final submission file

## Notes

- Test soundscapes are populated when running on Kaggle
- The submission notebook handles missing test files
- For better results, consider:
  - More training epochs
  - Whisper-based model for ensemble
  - Test-time augmentation
