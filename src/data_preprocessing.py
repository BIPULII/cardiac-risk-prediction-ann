import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data(filepath='data/heart.csv'):
    """Loads dataset, cleans it, splits it evenly, and applies scaling."""
    print("Booting up Data Preprocessing Pipeline...")
    
    # 1. Load Data
    df = pd.read_csv(filepath)
    
    # 2. Handle Missing Values (Drop the 6 rows with missing data)
    df_clean = df.dropna()
    
    # NEW: Save the explicitly cleaned dataset
    os.makedirs('outputs', exist_ok=True)
    df_clean.to_csv('outputs/cleaned_heart_data.csv', index=False)
    print(f"Cleaned dataset saved to outputs/cleaned_heart_data.csv (Dropped {len(df) - len(df_clean)} rows).")
    
    # 3. Separate features (X) and target (y)
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    
    # 4. Train/Test Split (80% training, 20% testing)
    # NEW: stratify=y ensures balanced class distribution in both splits!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Normalize/Scale numerical values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save the scaler for the Streamlit UI and FastAPI
    joblib.dump(scaler, 'outputs/scaler.pkl')
    print(f"Data scaled and preprocessed successfully. Final Shape: {df_clean.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/heart.csv')