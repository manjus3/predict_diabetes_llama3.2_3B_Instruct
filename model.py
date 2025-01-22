# Import libraries (removed pandas)
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib

def load_and_process_data(filename):
    # Read data from CSV
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [list(map(float, row)) for row in reader]
    
    # Separate features and target
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    # Identify column indices for zero handling
    zero_cols = {
        'Glucose': headers.index('Glucose'),
        'BloodPressure': headers.index('BloodPressure'),
        'SkinThickness': headers.index('SkinThickness'),
        'Insulin': headers.index('Insulin'),
        'BMI': headers.index('BMI')
    }
    
    # Store calculated means
    mean_values = {}

    # Replace zeros with column means (excluding outcome column)
    for col_name, col_idx in zero_cols.items():
        col_data = [row[col_idx] for row in X]
        mean_val = sum(col_data) / len(col_data)
        mean_values[col_name] = mean_val
  
        for row in X:
            if row[col_idx] == 0:
                row[col_idx] = mean_val
                
    return np.array(X), np.array(y), headers[:-1], mean_values

# Load and process data
X, y, feature_names, mean_values = load_and_process_data('diabetes.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization (updated to use feature_names)
# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Feature Importance
sorted_idx = np.argsort(model.coef_[0])
plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_names)[sorted_idx], model.coef_[0][sorted_idx])
plt.title('Feature Importance from Logistic Regression')
plt.xlabel('Coefficient Value')
plt.savefig('feature_importance.png')
plt.close()

# Save model and scaler, and mean values
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(mean_values, 'mean_values.pkl')
