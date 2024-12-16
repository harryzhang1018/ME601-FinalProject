import torch
import torch.nn as nn

# from nn_effpos_pytorch import NeuralNetwork

# Define the neural network
class NeuralNetwork_sm(nn.Module):
    def __init__(self):
        super(NeuralNetwork_sm, self).__init__()
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
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 512)
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
class NN_InverseKin():
    def __init__(self):
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        # Load the model weights
        self.model = NeuralNetwork().to(self.device)
        self.model.load_state_dict(torch.load('nn_1214.pth'))
        self.model.eval()

    def predict(self, input):
        X = torch.tensor(input, dtype=torch.float32).to(self.device)
        predicted_output = self.model(X)
        return predicted_output.detach().cpu().numpy()

if __name__ == '__main__':

    # plot the whole data set with predicted result
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the data
    # Define the folder containing the data
    data_folder = 'data/eff_pos'

    # Number of training files to include (set n)
    n = 100  # Control how many files to include

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
    test_input = X[:]
    print('test input:', test_input)
    nnmodel = NN_InverseKin()
    pred_y = nnmodel.predict(test_input)
    print('predicted output:', pred_y)

    plt.figure(figsize=(16, 8))
    plt.subplot(4, 1, 1)
    plt.plot(y[:,0], label='actual')
    plt.plot(pred_y[:,0], label='predicted')
    plt.ylabel('Joint Angle 1')
    plt.legend(loc='upper right')
    # plt.title('Joint Angle 1')
    plt.subplot(4, 1, 2)
    plt.plot(y[:,1], label='actual')
    plt.plot(pred_y[:,1], label='predicted')
    plt.legend(loc='upper right')
    plt.ylabel('Joint Angle 2')
    plt.subplot(4, 1, 3)
    plt.plot(y[:,2], label='actual')
    plt.plot(pred_y[:,2], label='predicted')
    plt.legend(loc='upper right')
    plt.ylabel('Joint Angle 3')
    plt.subplot(4, 1, 4)
    plt.plot(y[:,3], label='actual')
    plt.plot(pred_y[:,3], label='predicted')
    plt.legend(loc='upper right')
    plt.ylabel('Joint Angle 4')
    plt.xlabel('Sample')
    plt.tight_layout()
    plt.show()