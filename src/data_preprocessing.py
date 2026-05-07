import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data(filepath='data/heart.csv'):
    """Loads dataset, cleans it, splits it, and applies scaling."""
    
    # 1. Load Data
    df = pd.read_csv(filepath)
    
    # 2. Handle Missing Values (Drop the 6 rows with missing data)
    df = df.dropna()
    
    # 3. Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 4. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Normalize/Scale numerical values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save the scaler for the Streamlit UI
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(scaler, 'outputs/scaler.pkl')
    print(f"Data cleaned (Shape: {df.shape}) and preprocessed successfully.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Removed the ../ so it looks inside the current project folder
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/heart.csv')