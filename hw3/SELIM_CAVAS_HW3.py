import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Single Layer Perceptron
class Perceptron:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w1 = np.random.randn(self.input_dim, self.output_dim)
        self.b1 = np.zeros((1, self.output_dim)) 
        self.out = None
        self.train_mse = None
        self.test_mse = None
        
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.z1
        return self.a1

    def backward(self, x, y, lr):
    
        z1 = self.a1 - y
        w1 = np.dot(x.T, z1)
        b1 = np.sum(z1, axis=0)
        
        self.w1 -= lr * w1
        self.b1 -= lr * b1
        
    def train(self, x, y, epochs, lr):
        for _ in range(epochs):
            out = self.forward(x)
            self.backward(x, y, lr)
            self.train_mse = np.mean((y - out)**2)
            

    def predict(self, x):
        out = self.forward(x)
        self.out = out
        self.test_mse = np.mean((test_r - out)**2)
        return out
    

# Multi-layer perceptron
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.bias1 = np.zeros((1, self.hidden_dim))
        self.weights2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.bias2 = np.zeros((1, self.output_dim))
        self.out = None
        self.train_mse = None
        self.test_mse = None

    def forward(self, x):
        # First layer
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)

        # Second layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.z2 

        return self.a2

    def backward(self, x, y, learning_rate):

        n = x.shape[0]

        delta2 = self.a2 - y
        d_weights2 = (1/n) * np.dot(self.a1.T, delta2)
        d_bias2 = (1/n) * np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.weights2.T) * sigmoid_derivative(self.a1)
        d_weights1 = (1/n) * np.dot(x.T, delta1)
        d_bias1 = (1/n) * np.sum(delta1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2

    def train(self, x, y, epochs, lr):
        for _ in range(epochs):
            out = self.forward(x)
            self.backward(x, y, lr)
            self.train_mse = np.mean((y - out)**2)

    def predict(self, x):
        out = self.forward(x)
        self.out = out
        self.test_mse = np.mean((test_r - out)**2)
        return out


# Load data and sort
train_df = pd.read_csv('train.csv').sort_values(by=['x'])
test_df = pd.read_csv('test.csv').sort_values(by=['x'])

train_x = train_df['x'].values.reshape(-1, 1)
train_r = train_df['r'].values.reshape(-1, 1)
test_x = test_df['x'].values.reshape(-1, 1)
test_r = test_df['r'].values.reshape(-1, 1)

# Train and evaluate models
perceptron = Perceptron(input_dim=train_x.shape[-1], output_dim=1)
mlp10 = MLP(input_dim=train_x.shape[-1], hidden_dim=10, output_dim=1)
mlp20 = MLP(input_dim=train_x.shape[-1], hidden_dim=20, output_dim=1)
mlp50 = MLP(input_dim=train_x.shape[-1], hidden_dim=50, output_dim=1)

models = [perceptron, mlp10, mlp20, mlp50]
num_hidden_units = [0, 10, 20, 50]

train_mse = []
test_mse = []


# Plot prediction results
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, model in enumerate(models):
    model.train(train_x, train_r, epochs=2000, lr=0.03)
    model.predict(test_x)

    train_mse.append(model.train_mse)
    test_mse.append(model.test_mse)

    axs[i].scatter(train_x, train_r, color='blue', alpha = 0.5, label='Training data')
    axs[i].scatter(test_x, test_r, color='red', alpha = 0.5, label='Testing data')
    axs[i].plot(test_x, model.out, color='green', label='Model prediction')
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