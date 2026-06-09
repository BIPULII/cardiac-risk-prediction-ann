import os
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from src.data_preprocessing import load_and_preprocess_data

print("Initializing Advanced Hyperparameter Tuning Engine...")
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
    
    # Evaluate accuracy and AUC
    probs = model.predict(X_test_scaled).ravel()
    predictions = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probs)
    
    # Record metrics
    results.append({
        "Model Version": name,
        "Hidden Layers": len(layers),
        "Neurons": ", ".join(map(str, layers)),
        "Dropout": "Yes" if dropout_rate > 0 else "No",
        "Accuracy": acc,
        "ROC-AUC": auc
    })

# 1. Execute the Document's Test Cases
build_and_evaluate("ANN Version 1", [16])
build_and_evaluate("ANN Version 2", [32, 16])
build_and_evaluate("ANN Version 3", [64, 32], dropout_rate=0.2)

# 2. Generate the Professional Report
results_df = pd.DataFrame(results)

# Create a formatted version for the terminal and CSV
formatted_df = results_df.copy()
formatted_df['Accuracy'] = formatted_df['Accuracy'].apply(lambda x: f"{x:.2%}")
formatted_df['ROC-AUC'] = formatted_df['ROC-AUC'].apply(lambda x: f"{x:.4f}")

print("\n--- Hyperparameter Tuning Report ---")
print(formatted_df.to_string(index=False))

# 3. Save the CSV securely
os.makedirs('outputs', exist_ok=True)
formatted_df.to_csv('outputs/hyperparameter_tuning.csv', index=False)
print("\nReport saved securely to outputs/hyperparameter_tuning.csv")

# 4. NEW: Generate the Presentation Visual
print("Generating tuning comparison chart...")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Melt the dataframe to plot Accuracy and AUC side-by-side
melted_df = results_df.melt(id_vars="Model Version", value_vars=["Accuracy", "ROC-AUC"], var_name="Metric", value_name="Score")

sns.barplot(data=melted_df, x="Model Version", y="Score", hue="Metric", palette=['#3498db', '#9b59b6'])
plt.title('ANN Performance across Different Architectures', fontsize=14, fontweight='bold')
plt.ylim(0.75, 1.0) # Zoom in to highlight differences
plt.legend(loc='lower right')

# Save in High-Resolution
plt.savefig('outputs/hyperparameter_tuning_chart.png', bbox_inches='tight', dpi=300)
print("Presentation Graph saved securely to outputs/hyperparameter_tuning_chart.png")