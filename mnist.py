from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
digits=load_digits()


# Features and labels
X, y = digits.data,digits.target


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a perceptron
print("Training a perceptron...")
perceptron = Perceptron(max_iter=1000, eta0=0.00001, random_state=42)
perceptron.fit(X_train, y_train)

# Predict and evaluate
y_pred = perceptron.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
