import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_and_preprocess_data

print("Booting Explainability Engine...")

# Load Data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# Train a proxy Random Forest to extract feature importance easily
print("Extracting feature importances...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# The 13 clinical features based on the UCI dataset standard
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Generate the Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Clinical Feature Importance (Drivers of Cardiac Risk)')
plt.xlabel('Impact on Prediction')
plt.ylabel('Clinical Feature')

# Save to outputs
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/feature_importance.png', bbox_inches='tight')
print("Explainability graph saved securely to outputs/feature_importance.png")