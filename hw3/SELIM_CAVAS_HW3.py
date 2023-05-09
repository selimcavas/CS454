import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_x = train_df['x'].values
train_r = train_df['r'].values
test_x = test_df['x'].values
test_r = test_df['r'].values

# Add bias term to inputs
train_x = train_x.reshape(-1, 1)
test_x = test_x.reshape(-1, 1)
train_x = np.concatenate((train_x, np.ones((len(train_x), 1))), axis=1)
test_x = np.concatenate((test_x, np.ones((len(test_x), 1))), axis=1)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Single-layer perceptron
class Perceptron:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim, 1)
    
    def train(self, x, y, lr):
        y_pred = np.dot(x, self.weights) 
        mse = np.mean((y - y_pred)**2)
        
        error = y.reshape(-1, 1) - y_pred
        delta = error

        self.weights += lr * np.dot(x.reshape(-1, 1), delta)
            
        return mse
    
    def predict(self, x):
        return np.dot(x, self.weights)


# Multi-layer perceptron
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.random.randn(1, hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.random.randn(1, output_dim)
    
    def train(self, x, y, lr):
        # Forward pass
        hidden_layer = sigmoid(np.dot(x, self.weights1) + self.bias1)
        y_pred = np.dot(hidden_layer, self.weights2) + self.bias2
        mse = np.mean((y - y_pred)**2)

        # Backward pass
        error = y.reshape(-1,1) - y_pred
        delta = error * sigmoid_derivative(y_pred)
        hidden_error = np.dot(delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
        self.weights2 += lr * np.dot(hidden_layer.T, delta)
        self.bias2 += lr * np.sum(delta, axis=0, keepdims=True)
        self.weights1 += lr * np.dot(x.T, hidden_delta)
        self.bias1 += lr * np.sum(hidden_delta, axis=0, keepdims=True)

        return mse

    
    def predict(self, x):
        hidden_layer = sigmoid(np.dot(x, self.weights1) + self.bias1)
        return np.dot(hidden_layer, self.weights2) + self.bias2


# Train and evaluate models
perceptron = Perceptron(input_dim=train_x.shape[1])
mlp10 = MLP(input_dim=train_x.shape[1], hidden_dim=10, output_dim=1)
mlp20 = MLP(input_dim=train_x.shape[1], hidden_dim=20, output_dim=1)
mlp50 = MLP(input_dim=train_x.shape[1], hidden_dim=50, output_dim=1)

models = [perceptron, mlp10, mlp20, mlp50]
num_hidden_units = [0, 10, 20, 50]

train_mse = []
test_mse = []

# Train models and record MSE
for model in models:
    train_mse_history = []
    for epoch in range(1000):
        train_mse_epoch = []
        for i in range(train_x.shape[0]):
            x = train_x[i].reshape(1,-1)
            y = train_r[i].reshape(1,-1)
            mse = model.train(x, y, lr=0.26)
            train_mse_epoch.append(mse)
        train_mse_history.append(np.mean(train_mse_epoch))
    train_mse.append(train_mse_history[-1])
    test_mse.append(np.mean((test_r - model.predict(test_x))**2))

# Plot results
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, model in enumerate(models):
    axs[i].scatter(train_x[:, 0], train_r, color='blue', alpha = 0.5, label='Training data')
    axs[i].scatter(test_x[:, 0], test_r, color='red', alpha = 0.5, label='Testing data')
    x_range = np.linspace(test_x[:, 0].min(), test_x[:, 0].max(), 100) # Generate 100 points between min and max x values to smooth out the line
    y_range = model.predict(np.concatenate((x_range.reshape(-1, 1), np.ones((len(x_range), 1))), axis=1))
    axs[i].plot(x_range, y_range, color='green', label='Model prediction')
    axs[i].legend()
    axs[i].set_title(f'Model {i+1} ({num_hidden_units[i]} hidden units)')
plt.show()

# Plot network complexity vs error
plt.plot(num_hidden_units, train_mse, label='Training MSE')
plt.plot(num_hidden_units, test_mse, label='Testing MSE')
plt.xlabel('Number of hidden units')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()

