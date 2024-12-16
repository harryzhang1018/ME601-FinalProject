import matplotlib.pyplot as plt

# Your data from each printed epoch:
epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
          1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
          2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
losses = [0.4702, 0.3430, 0.2569, 0.3370, 0.1604, 0.1658, 0.1273, 0.1045, 0.1047, 0.0798,
          0.0924, 0.0913, 0.0841, 0.0618, 0.0693, 0.0887, 0.0690, 0.0514, 0.0421, 0.0716,
          0.0526, 0.0430, 0.0449, 0.0378, 0.0451, 0.0316, 0.0497, 0.0353, 0.0343, 0.0246]

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel("Epoch",fontsize=12)
plt.ylabel("Loss", fontsize=12)
# Optionally add a grid for easier reading
plt.grid(True)

# Show the plot
plt.show()
