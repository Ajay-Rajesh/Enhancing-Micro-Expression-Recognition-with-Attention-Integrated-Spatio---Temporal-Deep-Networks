# Micro-Expression Recognition using 3D CNN

A deep learning project for **facial micro-expression recognition** using **PyTorch** and **3D Convolutional Neural Networks (3D CNNs)**. The project explores multiple architectures, including a custom 3D CNN, R3D-18, and an attention-based model for video classification.

## Features
- Custom PyTorch Dataset for video sequences
- 3D CNN implementation for spatio-temporal learning
- R3D-18 transfer learning model
- Attention-enhanced 3D CNN
- Mixed precision training (AMP)
- Early stopping and model checkpointing
- Class balancing and data augmentation

## Tech Stack
- Python
- PyTorch
- TorchVision
- NumPy
- PIL
- tqdm

## Project Structure
```
dataset/
 ├── train/
 ├── val/
 └── test/

models/
training/
checkpoints/
```

## Results
The project compares multiple 3D deep learning architectures for micro-expression recognition and includes recommendations for improving accuracy, robustness, and training efficiency.

## Future Improvements
- Better data augmentation
- Learning rate scheduling
- Test-time augmentation
- Attention visualization
- Hyperparameter tuning

## Author
Ajay R
