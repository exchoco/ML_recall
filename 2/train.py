import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

N_CLASSES = 10

def train(train_data_path):
    # Load training data
    train_df = pd.read_csv(train_data_path)

    # Separate features and labels
    X_train = train_df.drop(columns=['id', 'gt_label'])  # Drop the 'id' and 'gt_label' columns to get features
    y_train = train_df['gt_label']  # Use 'gt_label' column as labels

    # One-hot encode the labels
    y_train = tf.one_hot(y_train, N_CLASSES).numpy()

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print('this is the split x train shape ', X_train.shape)
    print('this is the split y train shape ', y_train.shape)


    # Define the model
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(64, activation='relu'),
        Dense(N_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    return model

def main(train_data_path, test_data_path, output_path):
    # Train the model
    model = train(train_data_path)

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Separate features (exclude the 'id' column)
    X_test = test_df.drop(columns=['id'])

    # Perform inference with the trained model
    y_pred = model.predict(X_test)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()  # Get the class with the highest probability

    # Save the inference results to the 'label' column of the test dataframe
    test_df['label'] = y_pred_classes

    # Save the updated test dataframe to a new CSV file
    test_df.to_csv(output_path, index=False)

    print(f"Inference completed and saved to '{output_path}'.")

# Example usage
if __name__ == "__main__":
    main('train_data.csv', 'test_data.csv', 'test_data_with_predictions.csv')
