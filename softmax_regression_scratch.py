import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ------------------------------------ 1. LOAD DATA AND EDA ----------------------------------

try:
    iris = pd.read_csv('Iris.csv')
except FileNotFoundError:
    print("Error: CSV file not found.")

iris.drop('Id', axis=1, inplace=True)                       # Drop the ID column if present           

X = iris.drop('Species', axis=1).values                     # Features (sepal length, sepal width, petal length, petal width)
y = iris['Species'].values.reshape(-1, 1)                   # Target (species: 0, 1, 2)   

class_names = iris['Species'].unique()                      # Get unique class names from original labels
class_names.sort() 

print(f"\nDataset shape: X={X.shape}, y={y.shape}")

encoder = OneHotEncoder(sparse_output=False)                # One-hot encode the target variable for the "from scratch" implementation 
y_one_hot = encoder.fit_transform(y)                        # This is necessary for the categorical cross-entropy loss function.
print(f"\nOne-hot encoded target shape: {y_one_hot.shape}")

X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = train_test_split(       # Split data into training and testing sets
    X, y, y_one_hot, test_size=0.2, random_state=42, stratify=y                             # Stratify ensures that the proportion of classes is maintained in both splits.
)

scaler = StandardScaler()                                   # Feature Scaling
X_train_scaled = scaler.fit_transform(X_train)              # Standardize numerical features for Gradient Descent to converge efficiently.
X_test_scaled = scaler.transform(X_test)

X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]                    # Add a bias (intercept) term to the scaled features for the "from scratch" model
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

print(f"\nProcessed Training data shape (with bias): {X_train_b.shape}, {y_train_one_hot.shape}")
print(f"Processed Testing data shape (with bias): {X_test_b.shape}, {y_test_one_hot.shape}")

# ------------------------------------ 2. MULTI-CLASS SOFTMAX LOGISTIC REGRESSION FROM SCRATCH ----------------------------------

def softmax(z):                                             # Softmax activation function. Converts a vector of arbitrary real values to a probability distribution.
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))    # Subtract max for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def hypothesis(X, theta):                                   # Computes the hypothesis (predicted probabilities) for softmax regression.
    return softmax(X @ theta)

def cost_function(X, y_one_hot, theta):                     # Computes the Categorical Cross-Entropy cost.
    m = len(y_one_hot)
    predictions = hypothesis(X, theta)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)    # Avoid log(0) by clipping predictions
    cost = (-1 / m) * np.sum(y_one_hot * np.log(predictions))
    return cost

def gradient_descent(X, y_one_hot, theta, learning_rate, n_iterations):     # Performs gradient descent to optimize theta for softmax regression.

    m = len(y_one_hot)
    cost_history = []

    for iteration in range(n_iterations):
        predictions = hypothesis(X, theta)
        errors = predictions - y_one_hot                    # (y_hat - y)
        gradient = (1 / m) * X.T @ errors                   # Gradient calculation
        theta = theta - learning_rate * gradient            # Update theta
        cost = cost_function(X, y_one_hot, theta)
        cost_history.append(cost)

        if iteration % (n_iterations // 10) == 0:
            print(f"Iteration {iteration}/{n_iterations}, Cost: {cost:.4f}")

    return theta, cost_history

# theta_scratch will have shape (num_features + 1, num_classes)
theta_scratch = np.zeros((X_train_b.shape[1], y_train_one_hot.shape[1]))            # Initialize parameters
learning_rate = 0.1                                                                 # Learning rate might need tuning
n_iterations = 1000

print(f"Initial theta (scratch) shape: {theta_scratch.shape}")
print(f"Initial cost (scratch): {cost_function(X_train_b, y_train_one_hot, theta_scratch):.4f}")

# Train the model
print("\nStarting Gradient Descent training...")
theta_optimized, cost_history = gradient_descent(X_train_b, y_train_one_hot, theta_scratch, learning_rate, n_iterations)

print(f"Optimized theta (scratch) shape: {theta_optimized.shape}")
print(f"Final cost (scratch): {cost_function(X_train_b, y_train_one_hot, theta_optimized):.4f}")

y_prob_scratch = hypothesis(X_test_b, theta_optimized)                              # Make predictions on the test set
y_pred_scratch = np.argmax(y_prob_scratch, axis=1)                                  # Convert probabilities to class labels (index of the max probability)


# ------------------------------------ 3. EVALUATIONS FOR THE SCRATCH MODEL ----------------------------------


label_encoder = LabelEncoder()                              # Convert string class names to integers (0, 1, 2) to match prediction
y_test_int = label_encoder.fit_transform(y_test.ravel())    # Flatten y_test and encode

accuracy_s = accuracy_score(y_test_int, y_pred_scratch)
conf_matrix_s = confusion_matrix(y_test_int, y_pred_scratch)
class_report_s = classification_report(y_test_int, y_pred_scratch, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy_s:.4f}")
print("\nConfusion Matrix (From Scratch):")
print(conf_matrix_s)
print("\nClassification Report (From Scratch):")
print(class_report_s)

# ------------------------------------ 4. PLOTTING ----------------------------------

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(range(n_iterations), cost_history)
plt.title('Cost Function History (From Scratch)')
plt.xlabel('Iterations')
plt.ylabel('Cost (Categorical Cross-Entropy)')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_s, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (From Scratch)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# ------------------------------------ 5. METRICS INTERPRETATION ----------------------------------

print(f"Accuracy ({accuracy_s:.4f}): Proportion of total predictions that were correct across all classes.")

