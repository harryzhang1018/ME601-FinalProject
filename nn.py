import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn as nn
import torch.optim as optim

data_list = []

base_directory = './'#data/eff_pos' #TODO CHANGE TO ACTUAL FILE PATH 

for file_num in range(1, 1001): 
    file_path = os.path.join(base_directory, f"exp_{file_num}.csv")
    
    with open(file_path, 'r') as file:
        data = list(map(float, file.read().strip().split('\n'))) 
        data_list.append(data)

data = np.column_stack(data_list)

ee_poses = (data[0:3, :]).transpose()
arm_angles = (data[3:7, :]).transpose()

ee_poses = torch.tensor(ee_poses, dtype=torch.float32)
arm_angles = torch.tensor(arm_angles, dtype=torch.float32)

angles_train, angles_temp, ee_train, ee_temp = train_test_split(arm_angles, ee_poses, test_size=0.4, random_state=42)
angles_test, angles_val, ee_test, ee_val = train_test_split(angles_temp, ee_temp, test_size=0.5, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
angles_train_poly = poly.fit_transform(angles_train.numpy())
angles_val_poly = poly.transform(angles_val.numpy())
angles_test_poly = poly.transform(angles_test.numpy())

angles_train_poly = torch.tensor(angles_train_poly, dtype=torch.float32)
angles_val_poly = torch.tensor(angles_val_poly, dtype=torch.float32)
angles_test_poly = torch.tensor(angles_test_poly, dtype=torch.float32)


class RobotArmNN(nn.Module):
    def __init__(self):
        super(RobotArmNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),  # Increase neurons in the first layer
            nn.SiLU(),  # Use Swish activation
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),  # Deeper and wider layers
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )


    
    def forward(self, x):
        return self.network(x)
    
 #left the training stuff in here just in case   

model = RobotArmNN()
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
learning_rate = 0.001

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(angles_train.size(0)) 
    epoch_loss = 0
    for i in range(0, angles_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = angles_train[indices], ee_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(angles_val)
        val_loss = criterion(val_outputs, ee_val).item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

model.eval()

with torch.no_grad():
    test_outputs = model(angles_test)
    test_loss = criterion(test_outputs, ee_test).item()
    print(f"Test Loss: {test_loss:.4f}")

    predictions = test_outputs.numpy()
    actuals = ee_test.numpy()


residuals = predictions - actuals

fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

coordinates = ['x', 'y', 'z']
for i, ax in enumerate(axes):
    ax.scatter(actuals[:, i], residuals[:, i], alpha=0.7, label=f"{coordinates[i]}-coordinate Residuals")
    ax.axhline(0, color='r', linestyle='--', label="Perfect Prediction")
    ax.set_ylabel("Residuals (Prediction - Actual)")
    ax.set_title(f"{coordinates[i]}-coordinate Residual Plot")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Actual Values")
plt.tight_layout()
plt.show()
