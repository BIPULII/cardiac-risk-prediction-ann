import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
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

# 3. Compile and Train (NEW: Added AUC Metric for medical standard evaluation)
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Early stopping prevents overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Training Model...")
history = model.fit(
    X_train_scaled, y_train, 
    validation_data=(X_test_scaled, y_test), 
    epochs=100, 
    batch_size=16, 
    callbacks=[early_stop]
)

# 4. Save the Model securely
os.makedirs('models', exist_ok=True)
model.save('models/heart_ann_model.keras')
print("Model saved to models/heart_ann_model.keras")

# 5. NEW: Generate Presentation-Ready High-Resolution Graphs
print("Generating high-resolution training graphs...")
os.makedirs('outputs', exist_ok=True)

sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#3498db', linewidth=2.5)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linewidth=2.5)
ax1.set_title('Model Accuracy Progression', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xlabel('Training Epoch', fontsize=12)
ax1.legend(loc='lower right')

# Subplot 2: Loss
ax2.plot(history.history['loss'], label='Train Loss', color='#3498db', linewidth=2.5)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2.5)
ax2.set_title('Model Loss (Error) Reduction', fontsize=14, fontweight='bold')
ax2.set_ylabel('Binary Crossentropy Loss', fontsize=12)
ax2.set_xlabel('Training Epoch', fontsize=12)
ax2.legend(loc='upper right')

# Save the polished graph
plt.tight_layout()
plt.savefig('outputs/training_history.png', bbox_inches='tight', dpi=300)
print("Presentation-ready graph saved to outputs/training_history.png")