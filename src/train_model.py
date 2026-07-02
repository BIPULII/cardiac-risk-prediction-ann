import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score, 
    average_precision_score, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
)
from src.data_preprocessing import load_and_preprocess_data

# 1. Load data using Member 1's pipeline
print("Fetching preprocessed data...")
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# 2. Define the Optimized Architecture
model = Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'), 
    Dropout(0.2), 
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 3. Compile and Train (NEW: Added AUC Metric for medical standard evaluation)
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Early stopping prevents overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Training Model...")
history = model.fit(
    X_train_scaled, y_train, 
    validation_data=(X_test_scaled, y_test), 
    epochs=100, 
    batch_size=16, 
    callbacks=[early_stop]
)

# 4. Comprehensive Evaluation on Test Set
print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

# Get predictions and probabilities
y_pred_proba = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate all metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = recall_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Create evaluation report
evaluation_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall (Sensitivity)": sensitivity,
    "Specificity": specificity,
    "F1-Score": f1_score(y_test, y_pred),
    "Balanced Accuracy": balanced_acc,
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc,
    "Matthews Corr. Coeff": mcc,
    "Cohen's Kappa": kappa
}

print("\nMetric Scores:")
for metric, value in evaluation_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nConfusion Matrix:")
print(f"  True Negatives:  {tn}  |  False Positives: {fp}")
print(f"  False Negatives: {fn}  |  True Positives:  {tp}")

# Save evaluation metrics to CSV
import pandas as pd
eval_df = pd.DataFrame([evaluation_metrics])
os.makedirs('outputs', exist_ok=True)
eval_df.to_csv('outputs/model_evaluation_metrics.csv', index=False)
print("\nEvaluation metrics saved to outputs/model_evaluation_metrics.csv")

# 5. Save the Model securely
os.makedirs('models', exist_ok=True)
model.save('models/heart_ann_model.keras')
print("Model saved to models/heart_ann_model.keras")

# 6. NEW: Generate Presentation-Ready High-Resolution Graphs
print("Generating high-resolution training graphs...")
os.makedirs('outputs', exist_ok=True)

sns.set_theme(style="whitegrid")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#3498db', linewidth=2.5)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linewidth=2.5)
ax1.set_title('Model Accuracy Progression', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xlabel('Training Epoch', fontsize=12)
ax1.legend(loc='lower right')

# Subplot 2: Loss
ax2.plot(history.history['loss'], label='Train Loss', color='#3498db', linewidth=2.5)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2.5)
ax2.set_title('Model Loss (Error) Reduction', fontsize=14, fontweight='bold')
ax2.set_ylabel('Binary Crossentropy Loss', fontsize=12)
ax2.set_xlabel('Training Epoch', fontsize=12)
ax2.legend(loc='upper right')

# Subplot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False,
            xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_xlabel('Predicted Label', fontsize=12)

# Subplot 4: ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax4.plot(fpr, tpr, color='#2ecc71', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate', fontsize=12)
ax4.set_ylabel('True Positive Rate', fontsize=12)
ax4.set_title('ROC-AUC Curve', fontsize=14, fontweight='bold')
ax4.legend(loc="lower right")

# Save the polished graph
plt.tight_layout()
plt.savefig('outputs/training_evaluation.png', bbox_inches='tight', dpi=300)
print("Comprehensive evaluation graph saved to outputs/training_evaluation.png")