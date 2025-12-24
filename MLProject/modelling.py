import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Load Data (Train dan Test sudah terpisah dari preprocessing)
print("Loading preprocessed dataset...")
train_df = pd.read_csv('dataset_preprocessing/train_data.csv')
test_df = pd.read_csv('dataset_preprocessing/test_data.csv')

# Kolom target
TARGET_COLUMN = 'Heart Disease'

# Separate features and target
X_train = train_df.drop([TARGET_COLUMN], axis=1)
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop([TARGET_COLUMN], axis=1)
y_test = test_df[TARGET_COLUMN]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution:\n{y_train.value_counts()}")
print(f"Test class distribution:\n{y_test.value_counts()}")

# Enable autologging - MLflow run handles the experiment/run context
mlflow.sklearn.autolog()

# Train model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision (weighted): {precision:.4f}")
print(f"  Recall (weighted): {recall:.4f}")
print(f"  F1-Score (weighted): {f1:.4f}")

# Save model ke folder model/ untuk Docker
MODEL_DIR = 'model'
if os.path.exists(MODEL_DIR):
    import shutil
    shutil.rmtree(MODEL_DIR)
mlflow.sklearn.save_model(rf_model, MODEL_DIR)
print(f"\nModel (MLflow format) saved to: {MODEL_DIR}/")

print("\nModel berhasil dilatih!")