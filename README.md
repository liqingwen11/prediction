# prediction
height prediction and pattern recongnition from picture captured by high speed camera 
# Liquid Height Prediction

This project uses deep learning to predict the liquid height from images using PyTorch.

## Structure

- `dataset/` - Directory for your training and validation images.
- `models/` - Contains model definitions and the best trained model.
- `utils/` - Contains utility functions for data preprocessing.
- `train.py` - Script to train the model.
- `predict.py` - Script to predict the liquid height from a new image.

## Setup

1. Install dependencies:
   ```bash
   pip install torch torchvision pillow
