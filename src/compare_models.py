import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) # NEW METRIC
    })

# 3. Evaluate Your Existing ANN
print("Loading trained ANN...")
ann_model = tf.keras.models.load_model('models/heart_ann_model.keras')

# ANN predict() returns probabilities by default, perfect for AUC!
ann_probs = ann_model.predict(X_test_scaled).ravel() 
ann_predictions = (ann_probs > 0.5).astype(int)

results.append({
    "Model": "ANN (Yours)",
    "Accuracy": accuracy_score(y_test, ann_predictions),
    "Precision": precision_score(y_test, ann_predictions),
    "Recall": recall_score(y_test, ann_predictions),
    "F1-Score": f1_score(y_test, ann_predictions),
    "ROC-AUC": roc_auc_score(y_test, ann_probs) # NEW METRIC
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
print("Generating visual comparison graph...")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Melt the dataframe so Seaborn can plot it easily side-by-side
df_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Draw the bar chart
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
plt.title("Algorithm Performance Comparison", fontsize=14, fontweight='bold')
plt.ylim(0.75, 1.0) # Zoom in on the top 25% to make differences visible
plt.legend(loc='lower right')

# Save the graph
plt.savefig('outputs/model_comparison_chart.png', bbox_inches='tight')
print("Presentation Graph saved to outputs/model_comparison_chart.png")