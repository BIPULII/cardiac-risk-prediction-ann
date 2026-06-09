import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Initialize Pipeline
print("Booting up Data Analysis Engine...")
os.makedirs('outputs', exist_ok=True)
df = pd.read_csv('data/heart.csv')

# Set professional visual style
sns.set_theme(style="whitegrid")

# 2. Heart Disease Count Graph
print("Generating Disease Count graph...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='target', palette='Set2')
plt.title('Heart Disease Distribution (0 = No Disease, 1 = Disease)')
plt.savefig('outputs/eda_disease_count.png', bbox_inches='tight')
plt.close()

# 3. Age vs Heart Disease
print("Generating Age vs Disease graph...")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='target', multiple='stack', palette='Set2', kde=True)
plt.title('Age vs Heart Disease Risk')
plt.savefig('outputs/eda_age_vs_disease.png', bbox_inches='tight')
plt.close()

# 4. Cholesterol vs Heart Disease
print("Generating Cholesterol vs Disease graph...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='target', y='chol', palette='Set2')
plt.title('Cholesterol Levels by Heart Disease Status')
plt.savefig('outputs/eda_cholesterol_vs_disease.png', bbox_inches='tight')
plt.close()

# 5. Blood Pressure vs Heart Disease
print("Generating Blood Pressure vs Disease graph...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='target', y='trestbps', palette='Set2')
plt.title('Resting Blood Pressure by Heart Disease Status')
plt.savefig('outputs/eda_blood_pressure_vs_disease.png', bbox_inches='tight')
plt.close()

# 6. Correlation Heatmap
print("Generating Correlation Heatmap...")
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Clinical Feature Correlation Heatmap')
plt.savefig('outputs/eda_correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("\nSuccess! All 5 EDA graphs have been saved to the outputs/ directory.")