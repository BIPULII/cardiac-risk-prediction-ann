import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    roc_curve, auc
)
from src.data_preprocessing import load_and_preprocess_data
import tensorflow as tf

print("Fetching pipeline data...")
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# 1. Initialize Traditional ML Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# 2. Train and Evaluate Traditional Models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # Get probabilities for the ROC-AUC score
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 
    
    # Calculate detailed metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Specificity": specificity,
        "F1-Score": f1_score(y_test, y_pred),
        "Balanced Accuracy": balanced_acc,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Matthews Corr. Coeff": mcc,
        "Cohen's Kappa": kappa
    })

# 3. Evaluate Your Existing ANN
print("Loading trained ANN...")
ann_model = tf.keras.models.load_model('models/heart_ann_model.keras')

# ANN predict() returns probabilities by default, perfect for AUC!
ann_probs = ann_model.predict(X_test_scaled).ravel() 
ann_predictions = (ann_probs > 0.5).astype(int)

# Calculate detailed metrics for ANN
tn_ann, fp_ann, fn_ann, tp_ann = confusion_matrix(y_test, ann_predictions).ravel()
specificity_ann = tn_ann / (tn_ann + fp_ann)
sensitivity_ann = recall_score(y_test, ann_predictions)
balanced_acc_ann = balanced_accuracy_score(y_test, ann_predictions)
pr_auc_ann = average_precision_score(y_test, ann_probs)
mcc_ann = matthews_corrcoef(y_test, ann_predictions)
kappa_ann = cohen_kappa_score(y_test, ann_predictions)
roc_auc_ann = roc_auc_score(y_test, ann_probs)

results.append({
    "Model": "ANN (Yours)",
    "Accuracy": accuracy_score(y_test, ann_predictions),
    "Precision": precision_score(y_test, ann_predictions),
    "Recall": recall_score(y_test, ann_predictions),
    "Specificity": specificity_ann,
    "F1-Score": f1_score(y_test, ann_predictions),
    "Balanced Accuracy": balanced_acc_ann,
    "ROC-AUC": roc_auc_ann,
    "PR-AUC": pr_auc_ann,
    "Matthews Corr. Coeff": mcc_ann,
    "Cohen's Kappa": kappa_ann
})

# 4. Generate the Professional Text Report
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Report ---")
print(results_df.to_string(index=False))

# Ensure outputs folder exists
os.makedirs('outputs', exist_ok=True)

# Save the CSV
results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\nCSV Report saved to outputs/model_comparison.csv")

# 5. NEW: Generate a Beautiful Graph for the Presentation
print("Generating visual comparison graphs...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
sns.set_theme(style="whitegrid")

# Plot 1: Metrics Bar Chart
df_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
metrics_to_plot = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
df_plot = df_melted[df_melted["Metric"].isin(metrics_to_plot)]
sns.barplot(data=df_plot, x="Metric", y="Score", hue="Model", palette="viridis", ax=ax1)
ax1.set_title("Key Metrics Comparison", fontsize=14, fontweight='bold')
ax1.set_ylim(0.7, 1.0)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_xlabel("Metric", fontsize=12)

# Plot 2: Advanced Metrics
advanced_metrics = ["Balanced Accuracy", "ROC-AUC", "PR-AUC", "Cohen's Kappa"]
df_advanced = df_melted[df_melted["Metric"].isin(advanced_metrics)]
sns.barplot(data=df_advanced, x="Metric", y="Score", hue="Model", palette="coolwarm", ax=ax2)
ax2.set_title("Advanced Performance Metrics", fontsize=14, fontweight='bold')
ax2.set_ylim(0.6, 1.0)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_xlabel("Metric", fontsize=12)

# Plot 3: ROC Curves for all models
# Logistic Regression ROC
models_dict = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
for model_name, model_obj in models_dict.items():
    model_obj.fit(X_train_scaled, y_train)
    y_prob = model_obj.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')

# ANN ROC
fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_probs)
roc_auc_ann = auc(fpr_ann, tpr_ann)
ax3.plot(fpr_ann, tpr_ann, lw=2.5, label=f'ANN (AUC = {roc_auc_ann:.3f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC-AUC Curves Comparison', fontsize=14, fontweight='bold')
ax3.legend(loc="lower right")

# Plot 4: Model Quality Score (Weighted Average of Key Metrics)
quality_scores = []
for idx, row in results_df.iterrows():
    # Weighted average: ROC-AUC (40%), F1 (30%), Balanced Accuracy (20%), Cohen's Kappa (10%)
    score = (row['ROC-AUC']*0.4 + row['F1-Score']*0.3 + row['Balanced Accuracy']*0.2 + row["Cohen's Kappa"]*0.1)
    quality_scores.append(score)
results_df['Quality Score'] = quality_scores

colors = ['#3498db', '#2ecc71', '#e74c3c']
ax4.barh(results_df['Model'], results_df['Quality Score'], color=colors)
ax4.set_xlim(0.7, 1.0)
ax4.set_xlabel('Composite Quality Score', fontsize=12)
ax4.set_title('Overall Model Quality (Weighted Average)', fontsize=14, fontweight='bold')
for i, v in enumerate(quality_scores):
    ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/comprehensive_model_comparison.png', bbox_inches='tight', dpi=300)
print("Comprehensive comparison graph saved to outputs/comprehensive_model_comparison.png")