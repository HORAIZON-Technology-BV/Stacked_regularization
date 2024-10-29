
# Multi-omics analysis via manifold mixing
This repository implements a stacked model architecture using ExtraTreesClassifier as a base model and conducted stratified shuffle split cross-validation setup. The model is designed to select top features from each of multi-omics data, followed by Bayesian optimization to fine-tune the model parameters for enhanced performance.


## Installation
Download the required libraries according to explicite_environment.txt and requirements.txt


## Features of main.py
- Stratified Shuffle Split Cross-Validation with 50 splits
- Stacked Model Architecture with ExtraTreesClassifier for both layer-0 and top model
- Feature Selection using SelectKBest with ANOVA F-value criterion
- Bayesian Optimization to tune hyperparameters for enhanced model accuracy
- ROC-AUC Score and ROC Curve calculations for each model run
