import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

# 3. Evaluate Your Existing ANN
print("Loading trained ANN...")
ann_model = tf.keras.models.load_model('models/heart_ann_model.keras')
ann_predictions = (ann_model.predict(X_test_scaled) > 0.5).astype(int)

results.append({
    "Model": "ANN (Yours)",
    "Accuracy": accuracy_score(y_test, ann_predictions),
    "Precision": precision_score(y_test, ann_predictions),
    "Recall": recall_score(y_test, ann_predictions),
    "F1-Score": f1_score(y_test, ann_predictions)
})

# 4. Generate the Professional Report
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Report ---")
print(results_df.to_string(index=False))

# Save for the final presentation
results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\nReport saved to outputs/model_comparison.csv")