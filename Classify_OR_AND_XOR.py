from sklearn.linear_model import Perceptron
import numpy as np

# Define the datasets
data = {
    "OR": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 1])
    },
    "AND": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 0, 0, 1])
    },
    "XOR": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 0])
    }
}

# Initialize a dictionary to store results
results = {}

# Train the perceptron and analyze results
for key, value in data.items():
    X = value["X"]
    y = value["y"]
    
    # Initialize and train perceptron
    perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    perceptron.fit(X, y)
    
    # Predict and store results
    predictions = perceptron.predict(X)
    results[key] = {
        "weights": perceptron.coef_,
        "bias": perceptron.intercept_,
        "predictions": predictions,
        "accuracy": perceptron.score(X, y)
    }

# Print the results
for key, result in results.items():
    print(f"\n--- {key} Dataset ---")
    print(f"Weights: {result['weights']}")
    print(f"Bias: {result['bias']}")
    print(f"Predictions: {result['predictions']}")
    print(f"Accuracy: {result['accuracy']:.2f}")
