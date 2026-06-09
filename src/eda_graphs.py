import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Initialize Pipeline
print("Booting up Data Analysis Engine...")
os.makedirs('outputs', exist_ok=True)

# LOAD THE CLEANED DATA (Connecting to Member 1's pipeline)
try:
    df = pd.read_csv('outputs/cleaned_heart_data.csv')
    print("Successfully loaded cleaned dataset.")
except FileNotFoundError:
    print("Warning: Cleaned data not found. Falling back to raw data.")
    df = pd.read_csv('data/heart.csv')

# Set professional visual style
sns.set_theme(style="whitegrid", palette="muted")

# Create a temporary column for human-readable plotting
df['Risk_Label'] = df['target'].map({0: 'No Risk (0)', 1: 'Disease Risk (1)'})

# 2. Heart Disease Count Graph
print("Generating Disease Count graph...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Risk_Label', hue='Risk_Label', palette=['#2ecc71', '#e74c3c'], legend=False)
plt.title('Patient Distribution by Cardiac Risk Status', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Number of Patients')
plt.savefig('outputs/eda_disease_count.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Age vs Heart Disease
print("Generating Age vs Disease graph...")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='Risk_Label', multiple='stack', palette=['#2ecc71', '#e74c3c'], kde=True)
plt.title('Age Distribution Segmented by Cardiac Risk', fontsize=14, fontweight='bold')
plt.xlabel('Patient Age (Years)')
plt.ylabel('Count')
plt.savefig('outputs/eda_age_vs_disease.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Cholesterol vs Heart Disease (Upgraded to a Violin Plot for better data science display)
print("Generating Cholesterol vs Disease graph...")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Risk_Label', y='chol', hue='Risk_Label', palette=['#2ecc71', '#e74c3c'], legend=False)
plt.title('Serum Cholesterol Levels by Cardiac Risk Status', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Cholesterol (mg/dl)')
plt.savefig('outputs/eda_cholesterol_vs_disease.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Blood Pressure vs Heart Disease
print("Generating Blood Pressure vs Disease graph...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Risk_Label', y='trestbps', hue='Risk_Label', palette=['#2ecc71', '#e74c3c'], legend=False)
plt.title('Resting Blood Pressure by Cardiac Risk Status', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Resting Blood Pressure (mmHg)')
plt.savefig('outputs/eda_blood_pressure_vs_disease.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. Correlation Heatmap
print("Generating Correlation Heatmap...")
plt.figure(figsize=(14, 10))
# Drop the temporary text column before calculating correlation
correlation_matrix = df.drop('Risk_Label', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Clinical Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.savefig('outputs/eda_correlation_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nSuccess! High-resolution (300 DPI) EDA graphs have been saved to the outputs/ directory.")