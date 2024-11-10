# Channel Estimation and Regression Analysis Using Machine Learning

This repository contains code for simulating channel estimation in wireless communication systems using various machine learning models. Additionally, a linear regression analysis is conducted to understand the behavior of nonlinear relationships within generated datasets. The project leverages different regression techniques, including K-Nearest Neighbors, Decision Trees, and a Neural Network for channel estimation, along with a detailed regression analysis for modeling nonlinear features.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Simulation Details](#simulation-details)
- [Machine Learning Models Used](#machine-learning-models-used)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction
In wireless communication systems, accurate channel estimation is critical for efficient data transmission and reception. This project demonstrates the use of machine learning models to predict channel parameters from simulated pilot symbols with noise and analyzes regression performance on synthetic nonlinear data.

## Dependencies
Make sure you have the following Python libraries installed:
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow` (for the neural network model)

You can install the dependencies using:
```
pip install numpy scikit-learn matplotlib tensorflow
```
## Project Structure
- channel_estimation.py: Main code file for simulating channel estimation and regression analysis.
- README.md: This file, providing an overview and guide for the project.

## Simulation Details

1. Channel Estimation:
 - Simulation Parameters:
 - Number of samples: 5000
 - Number of pilot symbols: 10
 - Noise variance: 0.01
 - Pilot Symbols: Binary Phase Shift Keying (BPSK) modulation.
 - Channel Model: Rayleigh fading with added Gaussian noise.

2. Regression Analysis:
 - Simulates a nonlinear relationship with increased noise to challenge the regression model.

## Machine Learning Models Used

- K-Nearest Neighbors (KNN): Used for channel estimation, predicting the real and imaginary parts of the channel separately.
- Decision Tree: A simple tree-based model for estimating channels.
- Neural Network: A dense, multi-layer perceptron for complex predictions.
- Linear Regression: Analysis with a focus on modeling nonlinear relationships in synthetic data.

## Usage

- Clone the repository:
```
git clone https://github.com/yourusername/channel-estimation-ml.git
```
```
cd channel-estimation-ml
```
- Run the Python script:
```
python channel_estimation.py
```
## Results

- Channel Estimation: The models provide varying degrees of accuracy. The Neural Network typically yields lower mean squared error (MSE) compared to simpler models like KNN and Decision Trees.
- Regression Analysis: Linear regression performance is evaluated on nonlinear data, with visualizations provided to compare predicted and true values.

## Mean Squared Error (MSE)

- MSE values are displayed for each model in the output.
- Error distributions and scatter plots are generated for a comprehensive performance analysis.

## Plots

- The project generates several plots:

 - Channel Estimation: True vs. predicted values for both real and imaginary parts.
 - Error Distribution: Histogram of prediction errors.
 - Regression Analysis: Predicted vs. true values for the test set, with MSE displayed.

## Acknowledgements
The project uses concepts from wireless communications and machine learning. Special thanks to open-source communities for providing valuable resources.
