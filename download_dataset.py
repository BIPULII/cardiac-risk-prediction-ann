import pandas as pd
from ucimlrepo import fetch_ucirepo

# Download Heart Disease dataset from UCI
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Rename target column
df = df.rename(columns={"num": "target"})

# Convert target into binary classification
# 0 = no heart disease
# 1 = heart disease present
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# Save dataset
df.to_csv("data/heart.csv", index=False)

print("Dataset downloaded successfully.")
print(df.head())
print(df.shape)
print(df.columns)