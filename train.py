import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = "anxiety_attack_dataset.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Encode categorical features
label_columns = ["Smoking", "Family History of Anxiety", "Medication", "Recent Major Life Event", "Dizziness"]
one_hot_columns = ["Occupation"]

# Label encoding for binary or low cardinality columns (convert Yes/No to 1/0)
label_encoders = {}
for col in label_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].apply(lambda x: 1 if x == 'Yes' else 0))
    label_encoders[col] = le

# One-Hot Encoding for columns with many categories
df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

# Define features and target variable
X = df.drop(columns=["Severity of Anxiety Attack (1-10)"])
y = df["Severity of Anxiety Attack (1-10)"]

# Normalize numerical features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Normalize target variable (optional)
y_scaled = (y - 1) / 9  # Scale target to [0, 1]

# Save the scaler to a file for later use
joblib.dump(scaler, 'better_scaler.pkl')

# Set up K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_num = 1
fold_mae = []

# Define a simpler model
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['mae'])
    return model

# Cross-validation loop
for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_scaled[train_index], y_scaled[val_index]

    # Train the model
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, 
                        verbose=1, callbacks=[early_stopping])

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    fold_mae.append(mae)
    print(f"Fold {fold_num} - MAE: {mae}")
    fold_num += 1

# Print average MAE across all folds
print(f"Average MAE across all folds: {sum(fold_mae) / len(fold_mae)}")

# Final evaluation on test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
final_model = create_model()
final_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Evaluate the final model on test data
test_loss, test_mae = final_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the final model
final_model.save("better_anxiety_model.h5")
