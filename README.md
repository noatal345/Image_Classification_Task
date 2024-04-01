# Image Classification Project

This repository contains code for an image classification project aimed at classifying images into two categories: "faces" and "non-faces".
The project utilizes convolutional neural networks (CNNs) for training and evaluation.


## Files

- `analysis.py`: Contains functions for dataset analysis, visualization of misclassifications, calculation of performance metrics (precision, recall), and visualization of feature maps.
- `cnn_hyperparameter_tuning.py`: Script for hyperparameter tuning using Weights & Biases (wandb) configuration.
- `data.py`: Module for dataset loading, preprocessing, and analysis.
- `main.py`: Main script for running experiments with wandb integration.
- `model.py`: Definition of the CNN model architecture.
- `results`: Directory containing results such as misclassified images, feature map visualizations, wandb charts, and CSV files.
- `results.pptx`: PowerPoint presentation summarizing the project and results.
- `train_and_test.py`: Module containing functions for training and testing the CNN model.

## Usage
### Hyperparameter Tuning
Run the `cnn_hyperparameter_tuning.py to create a new sweep inside the wandb project.
### Analysis
To perform analysis on the trained model Run the `analysis.py (This script contains multiple functions to choose from)
