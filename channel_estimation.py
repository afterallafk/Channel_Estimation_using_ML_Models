import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression

# Parameters for simulation
num_samples = 5000  # Number of samples for channel estimation
num_pilots = 10     # Number of pilot symbols per sample
noise_variance = 0.01  # Noise variance for channel

# Generate random transmitted pilot symbols (BPSK)
transmitted_pilots = np.random.choice([-1, 1], size=(num_samples, num_pilots))

# Generate random true channels with Rayleigh fading
true_channels = np.random.normal(0, 1, size=(num_samples, num_pilots)) + \
                1j * np.random.normal(0, 1, size=(num_samples, num_pilots))

# Generate received pilots by applying channel and adding noise
received_pilots = true_channels * transmitted_pilots + np.sqrt(noise_variance) * (
    np.random.normal(0, 1, size=(num_samples, num_pilots)) +
    1j * np.random.normal(0, 1, size=(num_samples, num_pilots)))

# Prepare dataset: real and imaginary parts as features
X = np.hstack([np.real(received_pilots), np.imag(received_pilots)])
y = np.hstack([np.real(true_channels), np.imag(true_channels)])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define machine learning models for channel estimation
models = {
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=3),
    "Decision Tree": DecisionTreeRegressor(),
}

# Store predictions and metrics for channel estimation
predictions = {}
mse_values = {}

# Train and predict using each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions[model_name] = model.predict(X_test)
    # Calculate Mean Squared Error (MSE)
    mse_values[model_name] = mean_squared_error(y_test, predictions[model_name])

# Neural Network Model for Channel Estimation
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1])  # Output layer for real and imaginary parts
])

# Compile and train neural network
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
nn_predictions = nn_model.predict(X_test)
predictions["Neural Network"] = nn_predictions
mse_values["Neural Network"] = mean_squared_error(y_test, nn_predictions)

# Plot results for each model
for model_name, model_predictions in predictions.items():
    # Plot real part of channel estimation
    plt.figure(figsize=(10, 5))
    plt.plot(np.real(y_test[:, 0]), label='True Channel (Real)', linestyle='--')
    plt.plot(np.real(model_predictions[:, 0]), label=f'{model_name} Prediction (Real)')
    plt.title(f'Real Part of Channel Estimation - {model_name}')
    plt.legend()
    plt.show()

    # Plot imaginary part of channel estimation
    plt.figure(figsize=(10, 5))
    plt.plot(np.imag(y_test[:, 0]), label='True Channel (Imag)', linestyle='--')
    plt.plot(np.imag(model_predictions[:, 0]), label=f'{model_name} Prediction (Imag)')
    plt.title(f'Imaginary Part of Channel Estimation - {model_name}')
    plt.legend()
    plt.show()

    # Plot error distribution
    errors = y_test - model_predictions
    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=50, alpha=0.7)
    plt.title(f'Error Distribution - {model_name}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.show()

    # Print MSE for each model
    print(f"{model_name} MSE: {mse_values[model_name]}")

# Linear Regression Analysis with Nonlinear Relationships
np.random.seed(42)
num_samples = 5000
num_features = 5

# Generate random channel data with nonlinear relationships
X = np.random.rand(num_samples, num_features)

# Introduce a nonlinear relationship in the target variable
true_coefficients = np.array([1.5, -2.3, 0.7, 1.2, -0.8])
noise = np.random.normal(0, 0.2, num_samples)  # Increased noise
y = X @ true_coefficients + 0.5 * np.square(X[:, 0]) - 1.5 * np.sin(2 * np.pi * X[:, 1]) + noise

# Split the dataset into 70% training, 20% testing, and 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test and validation sets
y_pred = model.predict(X_test)
y_val_pred = model.predict(X_val)

# Calculate MSE for test and validation sets
test_mse = mean_squared_error(y_test, y_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# Print the results
print(f"Linear Regression - Test MSE: {test_mse:.4f}, Validation MSE: {val_mse:.4f}")

# Plot the predicted vs. true values for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"Linear Regression\nTest MSE: {test_mse:.4f}, Validation MSE: {val_mse:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()
