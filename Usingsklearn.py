from sklearn.linear_model import Perceptron
import numpy as np

# Define a linearly separable dataset
X = np.array([[2, 1], [1, -1], [-1, -1], [-2, 1]])  # Input features
y = np.array([1, 1, 0, 0])  # Labels

# Initialize the Perceptron
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# Train the Perceptron on the dataset
perceptron.fit(X, y)

# Make predictions
predictions = perceptron.predict(X)

# Print results
print("Weights:", perceptron.coef_)
print("Bias:", perceptron.intercept_)
print("Predictions:", predictions)
