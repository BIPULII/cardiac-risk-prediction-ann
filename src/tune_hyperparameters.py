import os
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from src.data_preprocessing import load_and_preprocess_data

print("Initializing Hyperparameter Tuning Engine...")
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('data/heart.csv')

results = []

def build_and_evaluate(name, layers, dropout_rate=0.0):
    print(f"\nTraining {name} (Neurons: {layers}, Dropout: {dropout_rate > 0})...")
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))
    
    # Dynamically build hidden layers
    for units in layers:
        model.add(Dense(units, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model (Silenced output for cleaner logs)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    
    # Evaluate accuracy
    predictions = (model.predict(X_test_scaled) > 0.5).astype(int)
    acc = accuracy_score(y_test, predictions)
    
    # Record metrics
    results.append({
        "Model Version": name,
        "Hidden Layers": len(layers),
        "Neurons": ", ".join(map(str, layers)),
        "Dropout": "Yes" if dropout_rate > 0 else "No",
        "Accuracy": f"{acc:.2%}"
    })

# 1. Execute the Document's Test Cases
build_and_evaluate("ANN Version 1", [16])
build_and_evaluate("ANN Version 2", [32, 16])
build_and_evaluate("ANN Version 3", [64, 32], dropout_rate=0.2)

# 2. Generate the Professional Report
results_df = pd.DataFrame(results)
print("\n--- Hyperparameter Tuning Report ---")
print(results_df.to_string(index=False))

# 3. Save to Outputs
os.makedirs('outputs', exist_ok=True)
results_df.to_csv('outputs/hyperparameter_tuning.csv', index=False)
print("\nReport saved securely to outputs/hyperparameter_tuning.csv")