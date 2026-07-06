import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from src.data_preprocessing import load_and_preprocess_data

print("Booting Evaluation Engine...")

# 1. Load Data using your existing pipeline
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# 2. Load the hyper-optimized Keras model
print("Loading ANN model...")
model = tf.keras.models.load_model('models/heart_ann_model.keras')

# 3. Generate Predictions on unseen test data
print("Generating predictions...")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------
# Artifact 1: Confusion Matrix
# ---------------------------------------------------------
print("Plotting Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('ANN Confusion Matrix - Cardiac Risk')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('outputs/confusion_matrix.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Artifact 2: ROC Curve (Receiver Operating Characteristic)
# ---------------------------------------------------------
print("Plotting ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - AI Model Performance')
plt.legend(loc="lower right")
plt.savefig('outputs/roc_curve.png', bbox_inches='tight')
plt.close()

print("Success! Enterprise evaluation graphs securely saved to the 'outputs/' directory.")