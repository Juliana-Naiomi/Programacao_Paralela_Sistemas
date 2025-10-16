"""
Definição das tarefas com diferentes prioridades
"""

import time
import numpy as np
from model import SimpleNeuralNetwork

class Task:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority
        
    def execute(self):
        raise NotImplementedError("Método execute deve ser implementado")
    
    def serialize(self):
        return {'name': self.name, 'priority': self.priority}
    
    @classmethod
    def deserialize(cls, data):
        raise NotImplementedError("Método deserialize deve ser implementado")

class TrainingTask(Task):
    def __init__(self, name, priority, data_size=1000, epochs=50):
        super().__init__(name, priority)
        self.data_size = data_size
        self.epochs = epochs
        
    def execute(self):
        # Simula tempo de processamento baseado na prioridade
        priority_delay = {"alta": 0.1, "media": 0.3, "baixa": 0.5}
        time.sleep(priority_delay.get(self.priority, 0.2))
        
        # Gera dados sintéticos
        X = np.random.randn(self.data_size, 10)
        y = np.random.randn(self.data_size, 1)
        
        # Cria e treina modelo
        model = SimpleNeuralNetwork(10, 5, 1)
        result = model.train(X, y, self.epochs)
        
        return f"{self.name} - {result}"
    
    def serialize(self):
        data = super().serialize()
        data.update({'data_size': self.data_size, 'epochs': self.epochs})
        return data
    
    @classmethod
    def deserialize(cls, data):
        return cls(data['name'], data['priority'], data['data_size'], data['epochs'])

class DataProcessingTask(Task):
    def __init__(self, name, priority, data_points=1000):
        super().__init__(name, priority)
        self.data_points = data_points
        
    def execute(self):
        # Simula tempo de processamento
        priority_delay = {"alta": 0.05, "media": 0.1, "baixa": 0.2}
        time.sleep(priority_delay.get(self.priority, 0.1))
        
        # Processamento de dados simulado
        data = np.random.randn(self.data_points, 20)
        processed_data = np.mean(data, axis=0)
        
        return f"{self.name} - Dados processados: {len(processed_data)} features"
    
    def serialize(self):
        data = super().serialize()
        data.update({'data_points': self.data_points})
        return data
    
    @classmethod
    def deserialize(cls, data):
        return cls(data['name'], data['priority'], data['data_points'])

class ModelEvaluationTask(Task):
    def __init__(self, name, priority, test_size=500):
        super().__init__(name, priority)
        self.test_size = test_size
        
    def execute(self):
        # Simula tempo de processamento
        priority_delay = {"alta": 0.08, "media": 0.15, "baixa": 0.25}
        time.sleep(priority_delay.get(self.priority, 0.1))
        
        # Avaliação de modelo simulado
        accuracy = np.random.uniform(0.85, 0.95)
        loss = np.random.uniform(0.1, 0.3)
        
        return f"{self.name} - Acurácia: {accuracy:.3f}, Loss: {loss:.3f}"
    
    def serialize(self):
        data = super().serialize()
        data.update({'test_size': self.test_size})
        return data
    
    @classmethod
    def deserialize(cls, data):
        return cls(data['name'], data['priority'], data['test_size'])