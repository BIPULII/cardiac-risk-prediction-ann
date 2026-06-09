import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from src.data_preprocessing import load_and_preprocess_data

# 1. Load data using Member 1's pipeline
print("Fetching preprocessed data...")
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# 2. Define the Optimized Architecture
model = Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'), 
    Dropout(0.2), 
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 3. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Training Model...")
history = model.fit(X_train_scaled, y_train, 
          validation_data=(X_test_scaled, y_test), 
          epochs=100, 
          batch_size=16, 
          callbacks=[early_stop])

# 4. Save the Model securely in the right folder
os.makedirs('models', exist_ok=True)
model.save('models/heart_ann_model.keras')
print("Model saved to models/heart_ann_model.keras")

# 5. Generate the Training Graph
os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('outputs/training_history.png')
print("Graph saved to outputs/training_history.png")