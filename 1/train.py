import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Load training data
train_df = pd.read_csv('train_data.csv')

# Separate features and labels
X_train = train_df.drop(columns=['id', 'gt_label'])  # Drop the 'id' and 'gt_label' columns to get features
y_train = train_df['gt_label']  # Use 'gt_label' column as labels

_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("this is X_train shape ", X_train.shape)
print("this is Y_train shape ", y_train.shape)

# Initialize and train the model (type 1- simple)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Initialize the model (type 2 - with GridSearchCV)###################################################
# model = RandomForestClassifier(random_state=42)

# Define the parameter grid (hyperparameters similar to learning rates, epochs, etc.)
param_grid = {
    'n_estimators': [100, 200, 500],  # Similar to the number of epochs
    'max_depth': [None, 10, 20, 30],  # Controls the depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Initialize GridSearchCV (like searching for the best learning rate or optimizer in neural networks)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV (like training the model with different hyperparameters)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")
model = best_model
######################################################################################################

y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Load test data
test_df = pd.read_csv('test_data.csv')

# Separate features (exclude the 'id' column)
X_test = test_df.drop(columns=['id'])
print('this is the x test shape ', X_test)

# Perform inference
y_pred = model.predict(X_test)
print('this is the y pred shape ', y_pred.shape)

# Save the inference results to the 'gt_label' column of the test dataframe
test_df['gt_label'] = y_pred

# Save the updated test dataframe to a new CSV file
test_df.to_csv('test_data_with_predictions.csv', index=False)

print("Inference completed and saved to 'test_data_with_predictions.csv'.")