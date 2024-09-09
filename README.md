# Stroke Prediction Using Autoencoders

## Project Overview

This project aims to predict the likelihood of a stroke using a dataset containing patient information, including demographic and clinical features. We leverage **three types of autoencoders**—Standard Autoencoder (AE), Denoising Autoencoder (DAE), and Variational Autoencoder (VAE)—to extract meaningful, compressed representations of the data. These representations are then used for classification tasks, predicting the risk of stroke.

## Autoencoders Used

1. **Standard Autoencoder (AE)**: Compresses high-dimensional input data into a lower-dimensional latent space and reconstructs the input from this compressed representation.
   - **Purpose**: Dimensionality reduction and feature extraction.
   
2. **Denoising Autoencoder (DAE)**: A variant of the Standard Autoencoder, trained to reconstruct the original data from a noisy version of the input.
   - **Purpose**: Handles noisy or incomplete data, improves robustness of feature learning.
   
3. **Variational Autoencoder (VAE)**: A probabilistic autoencoder that models the latent space as a distribution and introduces variability in the encoding process.
   - **Purpose**: Captures the uncertainty and variability in the data, with the ability to generate synthetic data.

## Dataset

- **Source**: The dataset used for this project is the Stroke Prediction dataset, which contains information about patients, such as age, gender, hypertension status, heart disease, smoking habits, and more.
- **Target Variable**: The target is whether the patient had a stroke (`1`) or not (`0`).

## Workflow

1. **Data Preprocessing**:
   - Handling missing values.
   - One-hot encoding for categorical variables.
   - Normalizing numerical features.

2. **Autoencoder Models**:
   - **Standard Autoencoder (AE)**: Trained on the input data for dimensionality reduction.
   - **Denoising Autoencoder (DAE)**: Trained on noisy input data to learn robust feature representations.
   - **Variational Autoencoder (VAE)**: Trained to model the latent space as a distribution, capturing the underlying variability in the data.

3. **Feature Extraction**:
   - Encoded features from each autoencoder are extracted.
   - These features are used as input for a classifier (Random Forest) for stroke prediction.

4. **Ensemble of Autoencoders**:
   - Features from all three autoencoders are combined to create an **ensemble** of autoencoders.
   - The combined feature set is used for classification to leverage the strengths of each autoencoder.

5. **Evaluation Metrics**:
   - The models are evaluated based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

## Installation and Setup

### Requirements

- **Python 3.x**
- **TensorFlow** (>= 2.0)
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

### Installation

You can install the required Python libraries using `pip`:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
