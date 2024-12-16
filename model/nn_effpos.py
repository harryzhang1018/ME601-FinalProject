import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Add this line to disable GPU
import numpy as np
import pandas as pd
from keras_core.models import Sequential
from keras_core.layers import BatchNormalization, Dropout, Dense
from keras_core.optimizers import Adam
from sklearn.preprocessing import StandardScaler  # Add this import

# Define the folder containing the data
data_folder = 'data/eff_pos'

# Number of training files to include (set n)
n = 1000  # Control how many files to include

# Read and concatenate specified CSV files
data = []
for i in range(1, n + 1):
    filename = f'{data_folder}/exp_{i}.csv'
    data_temp = np.genfromtxt(filename, delimiter='')
    data.append(data_temp)
data = np.vstack(data)

# Split the data into input (X) and output (y)
X = data[:, :3]  # Rows 1-3 are x, y, z (inputs), transpose to match shape
y = data[:, 3:]  # Rows 4-7 are theta_1 to theta_4 (outputs), transpose to match shape


# # Normalize the data for better performance
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X = scaler_X.fit_transform(X)
# y = scaler_y.fit_transform(y)

# Split into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the neural network
model = Sequential()
model.add(Dense(512, input_dim=3, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X, y, epochs=1000, batch_size=32, validation_data=(X, y), shuffle=True)

loss, mae = model.evaluate(X_test, y_test)

print('Test loss:', loss)
print('Test MAE:', mae)

# Function to denormalize predictions
def denormalize(data, scaler):
    return scaler.inverse_transform(data)

# Example prediction
sample_input = X_test[0:1]  # Use the first test sample
predicted_output = model.predict(sample_input)
actual_output = y_test[0:1]
# predicted_output = denormalize(predicted_output, scaler_y)  # Denormalize
# actual_output = denormalize(actual_output, scaler_y)

print("Predicted Output (theta_1 to theta_4):", predicted_output)
print("Actual Output (theta_1 to theta_4):", actual_output)
