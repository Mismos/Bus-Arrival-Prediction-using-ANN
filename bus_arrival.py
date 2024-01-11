import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

# Step 1: Data Collection
# Assuming you have a CSV file containing the historical bus arrival data
data = pd.read_csv('bus_data.csv')

# Step 2: Data Preprocessing
# Perform any necessary data cleaning and preprocessing steps
data = data.dropna()  # Drop rows with missing values

# Scale numerical features
scaler = MinMaxScaler()
data[['time_of_day']] = scaler.fit_transform(data[['time_of_day']])

# Perform one-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['weather_condition'])

# Step 3: Feature Selection
# Select relevant features for prediction, e.g., time of day, weather conditions
feature_cols = ['time_of_day', 'weather_condition_sunny', 'weather_condition_rainy', 'weather_condition_cloudy']

# Step 4: Data Splitting
# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# Step 5: Model Architecture
num_features = len(feature_cols)  # Number of input features
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer with predicted arrival time
])

# Step 6: Training
# Preprocess input features and target variable
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_data[feature_cols])
train_target = scaler.fit_transform(train_data['arrival_time'].values.reshape(-1, 1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_features, train_target, epochs=50, batch_size=32)

# Step 7: Validation
# Preprocess the testing data
test_features = scaler.transform(test_data[feature_cols])
test_target = scaler.transform(test_data['arrival_time'].values.reshape(-1, 1))

# Evaluate the model on the testing data
loss = model.evaluate(test_features, test_target)

# Step 8: Model Evaluation
# Convert the predicted arrival times back to the original scale
predicted_arrival_time = scaler.inverse_transform(model.predict(test_features))

# Convert the test target back to the original scale
test_target = scaler.inverse_transform(test_target)

# Calculate evaluation metrics for the original model
mae = mean_absolute_error(test_target, predicted_arrival_time)
mse = mean_squared_error(test_target, predicted_arrival_time)
rmse = np.sqrt(mse)

print("Evaluation Metrics (Original Model):")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Step 9: Prediction
# Given new input features, use the trained model to make predictions
new_features = np.array([[0.8, 1, 0, 0]])  # Example input features
scaled_features = scaler.transform(new_features)
predicted_arrival_time = scaler.inverse_transform(model.predict(scaled_features))

# Step 10: Save the Trained Model
model.save('bus_arrival_model.h5')

# Step 11: Load the Trained Model
loaded_model = tf.keras.models.load_model('bus_arrival_model.h5')

# Step 12: Prediction using the Loaded Model
new_features = np.array([[0.8, 1, 0, 0]])  # Example input features
scaled_features = scaler.transform(new_features)
predicted_arrival_time_loaded = scaler.inverse_transform(loaded_model.predict(scaled_features))

print("Predicted Arrival Time (Original Model):", predicted_arrival_time)
print("Predicted Arrival Time (Loaded Model):", predicted_arrival_time_loaded)
