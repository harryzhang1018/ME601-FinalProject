import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the folder containing the data
data_folder = 'data/eff_pos'

# Number of training files to include (set n)
n = 5000  # Control how many files to include

# Read and concatenate specified CSV files
data = []
for i in range(1, n + 1):
    filename = f'{data_folder}/exp_{i}.csv'
    data_temp = np.genfromtxt(filename, delimiter='')
    data.append(data_temp)
data = np.vstack(data)

# Split the data into input (X) and output (y)
X = data[:, :3]  # Rows 1-3 are x, y, z (inputs)
y = data[:, 3:]  # Rows 4-7 are theta_1 to theta_4 (outputs)

# # Normalize the data for better performance
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X = scaler_X.fit_transform(X)
# y = scaler_y.fit_transform(y)


# Convert to PyTorch tensors and move to device
X_train = torch.tensor(X, dtype=torch.float32).to(device)
X_test = torch.tensor(X, dtype=torch.float32).to(device)
y_train = torch.tensor(y, dtype=torch.float32).to(device)
y_test = torch.tensor(y, dtype=torch.float32).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 3000
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')

# Evaluation
model.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

# Load the model weights
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Example prediction after loading the model
sample_input = X_test[0:1]
predicted_output = model(sample_input)
actual_output = y_test[0:1]

print("Predicted Output (theta_1 to theta_4):", predicted_output.detach().cpu().numpy())
print("Actual Output (theta_1 to theta_4):", actual_output.detach().cpu().numpy())