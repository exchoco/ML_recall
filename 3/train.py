import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

def denormalize(data, min_value, max_value):
    return data * (max_value - min_value) + min_value

def train(train_data_path):
    # Load training data
    train_df = pd.read_csv(train_data_path)

    # Separate features and labels
    X_train = train_df.drop(columns=['id', 'gt_label']).astype(np.float32)  # Drop the 'id' and 'gt_label' columns to get features and convert to float32
    y_train = train_df['gt_label'].astype(np.float32)  # Use 'gt_label' column as labels and convert to float32

    # Optional: Normalize features (0 to 1) and labels (0 to 1)
    X_train_min, X_train_max = X_train.min(), X_train.max()
    y_train_min, y_train_max = y_train.min(), y_train.max()
    
    X_train = normalize(X_train, X_train_min, X_train_max)
    y_train = normalize(y_train, y_train_min, y_train_max)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),  # Automatically determine the number of features
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Single neuron output for regression
    ])

    # Compile the model with a specific learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    # Train the model and print the accuracy
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Print the training and validation accuracy for each epoch
    for epoch in range(10):
        train_mae = history.history['mae'][epoch]
        val_mae = history.history['val_mae'][epoch]
        print(f"Epoch {epoch+1}: Training MAE = {train_mae:.4f}, Validation MAE = {val_mae:.4f}")

    return model, X_train_min, X_train_max, y_train_min, y_train_max

def main(train_data_path, test_data_path, output_path):
    # Train the model
    model, X_train_min, X_train_max, y_train_min, y_train_max = train(train_data_path)

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Separate features (exclude the 'id' and 'gt_label' columns)
    X_test = test_df.drop(columns=['id']).astype(np.float32)  # Convert to float32

    # Optional: Normalize test features
    X_test = normalize(X_test, X_train_min, X_train_max)

    # Perform inference with the trained model
    y_pred = model.predict(X_test).flatten()  # Flatten the predictions to a 1D array

    # Optional: Denormalize predictions
    y_pred = denormalize(y_pred, y_train_min, y_train_max)

    # Save the inference results to the 'label' column of the test dataframe
    test_df['label'] = y_pred

    # Save the updated test dataframe to a new CSV file
    test_df.to_csv(output_path, index=False)

    print(f"Inference completed and saved to '{output_path}'.")

# Example usage
if __name__ == "__main__":
    main('train_data.csv', 'test_data.csv', 'test_data_with_predictions.csv')
