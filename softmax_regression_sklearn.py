import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------ 1. LOAD DATA AND EDA ----------------------------------

try:
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    print("Error: 'Iris.csv' not found.")
    exit()

print(df.head())

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']            # Define features (X) and target (y)
target_column = 'Species'

X = df[feature_columns]
y_raw = df[target_column]

label_encoder = LabelEncoder()                  # Encode target variable (species names) into numerical labels
y = label_encoder.fit_transform(y_raw)          # y will now be 0, 1, 2
iris_target_names = label_encoder.classes_      # Get the original class names for plotting

print(f"\nDataset shape: X={X.shape}, y={y.shape}")
print(f"Iris Species Names: {list(iris_target_names)}")

X_train, X_test, y_train, y_test = train_test_split(            # Split data into training and testing sets. Stratify ensures that the proportion of classes is maintained in both splits.
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()                                       # Feature Scaling
X_train_scaled = scaler.fit_transform(X_train)                  # Standardize numerical features. This is important for many scikit-learn solvers as well.
X_test_scaled = scaler.transform(X_test)

print(f"\nProcessed Training data shape (scaled): {X_train_scaled.shape}, {y_train.shape}")
print(f"Processed Testing data shape (scaled): {X_test_scaled.shape}, {y_test.shape}")

# ------------------------------------ 2. MULTI-CLASS SOFTMAX LOGISTIC REGRESSION FROM Scikit-learn ----------------------------------

model_sklearn = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model_sklearn.fit(X_train_scaled, y_train)

y_pred_sklearn = model_sklearn.predict(X_test_scaled)               # Make predictions on the scaled test set
y_prob_sklearn = model_sklearn.predict_proba(X_test_scaled)         # Probabilities for each class

# ------------------------------------ 3. EVALUATIONS FOR THE SKLEARN MODEL ----------------------------------

accuracy_s = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_s = confusion_matrix(y_test, y_pred_sklearn)
class_report_s = classification_report(y_test, y_pred_sklearn, target_names=iris_target_names)

print(f"Accuracy: {accuracy_s:.4f}")
print("\nConfusion Matrix (Scikit-learn):")
print(conf_matrix_s)
print("\nClassification Report (Scikit-learn):")
print(class_report_s)


# ------------------------------------ 4. PLOTTING ----------------------------------

plt.figure(figsize=(18, 7))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_s, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=iris_target_names, yticklabels=iris_target_names)
plt.title('Confusion Matrix (Scikit-learn)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plotting feature distribution for a single feature for one class. We'll use 'petal_length' as it often shows good separation
petal_length_idx = feature_columns.index('PetalLengthCm')        # Get index of 'petal_length'
plt.subplot(1, 2, 2)
sns.histplot(x=X_test.iloc[y_test == 0, petal_length_idx], color='red', label=iris_target_names[0], kde=True, stat='density', alpha=0.5)
sns.histplot(x=X_test.iloc[y_test == 1, petal_length_idx], color='green', label=iris_target_names[1], kde=True, stat='density', alpha=0.5)
sns.histplot(x=X_test.iloc[y_test == 2, petal_length_idx], color='blue', label=iris_target_names[2], kde=True, stat='density', alpha=0.5)
plt.title(f'Distribution of Petal Length by Species (Test Set)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------ 5. METRICS INTERPRETATION ----------------------------------

print(f"Accuracy ({accuracy_s:.4f}): Proportion of total predictions that were correct across all classes.")

