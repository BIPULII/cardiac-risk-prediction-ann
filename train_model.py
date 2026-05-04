import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib

# 1. Load Data (Piyumi's cleaned output)
# Replace 'cleaned_heart_data.csv' with your actual file
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1) # 'target' is 1 (disease) or 0 (no disease)
y = df['target']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the Data (Crucial for ANN convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so the Streamlit app can use it on new patient data
joblib.dump(scaler, 'scaler.pkl')

# 4. Define the ANN Architecture
# We use small dense layers and heavy Dropout (0.3) to prevent overfitting
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # Binary classification output
])

# 5. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training Model...")
# Early stopping prevents the model from memorizing the tiny dataset
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train_scaled, y_train, 
          validation_data=(X_test_scaled, y_test), 
          epochs=100, 
          batch_size=16, 
          callbacks=[early_stop])

# 6. Save the Model
model.save('cardiac_ann_model.h5')
print("Model and Scaler saved successfully.")