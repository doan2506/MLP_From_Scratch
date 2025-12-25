import numpy as np
import pandas as pd

HIDDEN_LAYER_COUNT = 2
HIDDEN_LAYER_SIZES = [128, 64]
OUTPUT_LAYER_SIZE = 1
CLASS_WEIGHTS = {0: 1.0, 1: 9.0}
BETAS = [0.9, 0.999]

class MLP:
    def __init__(self, X, y, learning_rate=0.001):
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy().reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        self.learning_rate = learning_rate
        self.W = []
        self.b = []
        self.beta1 = BETAS[0]
        self.beta2 = BETAS[1]
        self.epsilon = 1e-8
        self.t = 0
        self.m_W = []
        self.v_W = []
        self.m_b = []
        self.v_b = []

    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def weighted_binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - np.mean(CLASS_WEIGHTS[1] * y_true * np.log(y_pred) + CLASS_WEIGHTS[0] * (1 - y_true) * np.log(1 - y_pred))
    
    def loss_derivative(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (CLASS_WEIGHTS[1] * y_true / y_pred) + (CLASS_WEIGHTS[0] * (1 - y_true) / (1 - y_pred))
    
    def sigmoid_derivative(self, o):
        return o * (1 - o)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward_pass(self, X_batch):
        activations = [X_batch]
        nets = []
        layer_input = X_batch
        for i in range(HIDDEN_LAYER_COUNT + 1):
            if i < HIDDEN_LAYER_COUNT:
                z = np.dot(layer_input, self.W[i].T) + self.b[i]
                layer_output = self.relu(z)
            else:
                z = np.dot(layer_input, self.W[i].T) + self.b[i]
                layer_output = self.sigmoid(z)
            layer_input = layer_output
            activations.append(layer_output)
            nets.append(z)
        return layer_output, activations, nets

    def back_propagation(self, y_batch, activations, nets):
        delta_W = []
        delta_b = []
        batch_size = y_batch.shape[0]
        y_pred = activations[-1]
        dL_dz = self.loss_derivative(y_batch, y_pred) * self.sigmoid_derivative(y_pred)
        dL_dW = np.dot(activations[-2].T, dL_dz) / batch_size
        dL_db = np.sum(dL_dz, axis=0, keepdims=True) / batch_size
        delta_W.insert(0, dL_dW)
        delta_b.insert(0, dL_db)
        for i in range(HIDDEN_LAYER_COUNT - 1, -1, -1):
            dL_dz = np.dot(dL_dz, self.W[i+1]) * self.relu_derivative(nets[i])
            dL_dW = np.dot(activations[i].T, dL_dz) / batch_size
            dL_db = np.sum(dL_dz, axis=0, keepdims=True) / batch_size
            delta_W.insert(0, dL_dW)
            delta_b.insert(0, dL_db)
        self.t += 1
        for i in range(len(self.W)):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * delta_W[i].T
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (delta_W[i].T ** 2)
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            self.W[i] -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * delta_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (delta_b[i] ** 2)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.b[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


    def train(self, batch_size, epochs):
        for layer in range(HIDDEN_LAYER_COUNT):
            sigma = np.sqrt(2 / (self.X.shape[1] if layer == 0 else HIDDEN_LAYER_SIZES[layer - 1]))
            W_layer = np.random.randn(HIDDEN_LAYER_SIZES[layer], self.X.shape[1] if layer == 0 else HIDDEN_LAYER_SIZES[layer - 1]) * sigma
            self.W.append(W_layer)
            b_layer = np.zeros((1, HIDDEN_LAYER_SIZES[layer]))
            self.b.append(b_layer)
            self.m_W.append(np.zeros_like(W_layer))
            self.v_W.append(np.zeros_like(W_layer))
            self.m_b.append(np.zeros_like(b_layer))
            self.v_b.append(np.zeros_like(b_layer))
        W_output = np.random.randn(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES[-1]) * np.sqrt(1 / HIDDEN_LAYER_SIZES[-1])
        b_output = np.zeros((1, OUTPUT_LAYER_SIZE))
        self.W.append(W_output)
        self.b.append(b_output)
        self.m_W.append(np.zeros_like(W_output))
        self.v_W.append(np.zeros_like(W_output))
        self.m_b.append(np.zeros_like(b_output))
        self.v_b.append(np.zeros_like(b_output))
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            for start in range(0, self.X.shape[0], batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]
                y_pred, activations, nets = self.forward_pass(X_batch)
                batch_loss = self.weighted_binary_cross_entropy(y_batch, y_pred)
                epoch_loss += batch_loss * batch_size
                self.back_propagation(y_batch, activations, nets)
            epoch_loss /= self.X.shape[0]
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X):
        X_np = X.to_numpy()
        y_pred, _, _ = self.forward_pass(X_np)
        return (y_pred >= 0.5).astype(int)