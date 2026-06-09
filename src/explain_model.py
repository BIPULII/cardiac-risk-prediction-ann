import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_and_preprocess_data

print("Booting Advanced Explainability Engine...")

# Load Data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# Train a proxy Random Forest to extract feature importance easily
print("Extracting feature importances...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 1. NEW: Dictionary to translate raw column names to human-readable Presentation Labels
feature_mapping = {
    'age': 'Age',
    'sex': 'Sex (Gender)',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (Exercise)',
    'slope': 'Slope of Peak ST Segment',
    'ca': 'Major Vessels Colored by Fluoroscopy',
    'thal': 'Thalassemia Type'
}

raw_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Apply the mapping
readable_features = [feature_mapping[f] for f in raw_features]

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': readable_features,
    'Importance': rf_model.feature_importances_
})

# 2. NEW: Convert to percentages for easier audience comprehension
importance_df['Importance %'] = importance_df['Importance'] * 100
importance_df = importance_df.sort_values(by='Importance %', ascending=False)

# 3. NEW: Generate the High-Resolution Visualization
print("Generating presentation-ready graph...")
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Draw the barplot with a more professional color palette
ax = sns.barplot(x='Importance %', y='Feature', data=importance_df, palette='magma')

# Add the exact percentage text directly onto the end of each bar
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f%%', padding=5, fontweight='bold')

plt.title('Top Clinical Drivers of Cardiac Risk', fontsize=16, fontweight='bold')
plt.xlabel('Impact on AI Prediction (%)', fontsize=12)
plt.ylabel('') # Left blank because the feature names explain themselves

# Save to outputs in 300 DPI High-Resolution
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/feature_importance.png', bbox_inches='tight', dpi=300)
print("Presentation graph saved securely to outputs/feature_importance.png")