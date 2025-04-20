import numpy as np
import gzip
import urllib.request
import os
import logging
import dataPreparation as dp

# Load MNIST dataset
def load_mnist():
    test_labels, test_images, train_labels, train_images = dp.process()
    X_train = train_images
    y_train = train_labels
    return X_train[:100], y_train[:100]  # Use a subset for simplicity

# Sign activation function
def sign(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))

# Define the neural network
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.01

    def forward(self, X):
        self.hidden_layer = sign(np.dot(X, self.weights_input_hidden))
        self.output_layer = sign(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer

    def backward(self, X, y, learning_rate):
        output_error = y - self.output_layer
        hidden_error = output_error.dot(self.weights_hidden_output.T)

        # Update weights based on the error
        self.weights_hidden_output += self.hidden_layer.T.dot(output_error) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_error) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.output_layer))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Prepare the data
X_train, y_train = load_mnist()

# Flatten the input images from (100, 28, 28) to (100, 784)
X_train = X_train.reshape(X_train.shape[0], -1)

print(X_train.shape)  # Should be (100, 784)
print(y_train.shape)  # Should be (100,)

# One-hot encoding for the output labels
y_train_onehot = np.zeros((y_train.size, 10))
y_train_onehot[np.arange(y_train.size), y_train] = 1  # One-hot encoding

# Initialize and train the neural network
nn = SimpleNN(input_size=28 * 28, hidden_size=64, output_size=10)
nn.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X_train)
accuracy = np.mean(predictions == y_train)
print(f'Accuracy: {accuracy * 100:.2f}%')


