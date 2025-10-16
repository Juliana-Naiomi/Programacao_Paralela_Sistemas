"""
Modelo de rede neural simples para demonstração
"""

import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        
    def forward(self, X):
        self.hidden = np.tanh(np.dot(X, self.weights1))
        output = np.dot(self.hidden, self.weights2)
        return output
    
    # Isto NÃO é treinamento real de IA
    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Apenas um loop vazio - não aprende nada
            # Backward pass (simplificado)
            error = output - y
            d_weights2 = np.dot(self.hidden.T, error)
            d_weights1 = np.dot(X.T, np.dot(error, self.weights2.T) * (1 - np.power(self.hidden, 2)))
            
            # Update weights
            self.weights1 -= learning_rate * d_weights1
            self.weights2 -= learning_rate * d_weights2
            
        return f"Treinamento concluído - {epochs} epochs"