"""
Sistema de Treinamento de Modelos de IA com Processamento Paralelo
Usando MPI para distribuição de tarefas com prioridades
"""

from mpi4py import MPI
import time
import numpy as np
import argparse
import sys

# Configurar encoding para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class TrainingTask:
    def __init__(self, name, priority, data_size=1000, epochs=50):
        self.name = name
        self.priority = priority
        self.data_size = data_size
        self.epochs = epochs
        
    def execute(self):
        # Simula tempo de processamento baseado na prioridade
        priority_delay = {"alta": 0.1, "media": 0.3, "baixa": 0.5}
        time.sleep(priority_delay.get(self.priority, 0.2))
        
        # Gera dados sintéticos
        X = np.random.randn(self.data_size, 10)
        y = np.random.randn(self.data_size, 1)
        
        # Simula treinamento de modelo
        for epoch in range(self.epochs):
            # Processamento simulado
            pass
            
        return f"{self.name} - Treinamento concluido - {self.epochs} epocas"
    
    def serialize(self):
        return {
            'name': self.name, 
            'priority': self.priority,
            'data_size': self.data_size, 
            'epochs': self.epochs
        }
    
    @classmethod
    def deserialize(cls, data):
        return cls(data['name'], data['priority'], data['data_size'], data['epochs'])

class DataProcessingTask:
    def __init__(self, name, priority, data_points=1000):
        self.name = name
        self.priority = priority
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
        return {
            'name': self.name, 
            'priority': self.priority,
            'data_points': self.data_points
        }
    
    @classmethod
    def deserialize(cls, data):
        return cls(data['name'], data['priority'], data['data_points'])

class ModelEvaluationTask:
    def __init__(self, name, priority, test_size=500):
        self.name = name
        self.priority = priority
        self.test_size = test_size
        
    def execute(self):
        # Simula tempo de processamento
        priority_delay = {"alta": 0.08, "media": 0.15, "baixa": 0.25}
        time.sleep(priority_delay.get(self.priority, 0.1))
        
        # Avaliação de modelo simulado
        accuracy = np.random.uniform(0.85, 0.95)
        loss = np.random.uniform(0.1, 0.3)
        
        return f"{self.name} - Acuracia: {accuracy:.3f}, Loss: {loss:.3f}"
    
    def serialize(self):
        return {
            'name': self.name, 
            'priority': self.priority,
            'test_size': self.test_size
        }
    
    @classmethod
    def deserialize(cls, data):
        return cls(data['name'], data['priority'], data['test_size'])

class ParallelTrainingScheduler:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tasks_queue = []
        
    def create_tasks(self):
        """Cria uma lista de tarefas com diferentes prioridades"""
        tasks = [
            # Tarefas de ALTA prioridade (criticas)
            TrainingTask("model_critical", priority="alta", data_size=1000, epochs=50),
            DataProcessingTask("sensor_data", priority="alta", data_points=500),
            
            # Tarefas de MEDIA prioridade
            TrainingTask("model_secondary", priority="media", data_size=800, epochs=30),
            ModelEvaluationTask("model_validation", priority="media", test_size=200),
            DataProcessingTask("batch_processing", priority="media", data_points=300),
            
            # Tarefas de BAIXA prioridade
            TrainingTask("model_background", priority="baixa", data_size=600, epochs=20),
            ModelEvaluationTask("final_validation", priority="baixa", test_size=100),
            DataProcessingTask("archive_data", priority="baixa", data_points=150)
        ]
        return tasks
    
    def sort_tasks_by_priority(self, tasks):
        """Ordena tarefas por prioridade (alta > media > baixa)"""
        priority_order = {"alta": 0, "media": 1, "baixa": 2}
        return sorted(tasks, key=lambda x: priority_order[x.priority])
    
    def sequential_execution(self, tasks):
        """Execucao sequencial para comparacao de performance"""
        if self.rank == 0:
            print("\n" + "="*50)
            print("EXECUCAO SEQUENCIAL")
            print("="*50)
            
            start_time = time.time()
            results = []
            
            for task in tasks:
                print(f"Processando: {task.name} [{task.priority.upper()}]")
                result = task.execute()
                results.append(result)
                print(f"Concluido: {task.name} -> {result}")
            
            sequential_time = time.time() - start_time
            print(f"\nTempo total sequencial: {sequential_time:.4f} segundos")
            return sequential_time
        return 0
    
    def parallel_execution(self, tasks):
        """Execucao paralela distribuida entre processos"""
        if self.rank == 0:
            print("\n" + "="*50)
            print("EXECUCAO PARALELA")
            print("="*50)
            print(f"Distribuindo {len(tasks)} tarefas entre {self.size} processos...")
            
            # Ordena tarefas por prioridade
            sorted_tasks = self.sort_tasks_by_priority(tasks)
            
            # Distribui tarefas para processos (mestre)
            start_time = time.time()
            completed_tasks = 0
            total_tasks = len(sorted_tasks)
            
            # Envia tarefas iniciais para todos os workers
            for worker_rank in range(1, min(self.size, total_tasks + 1)):
                task_index = worker_rank - 1
                task_data = {
                    'task_index': task_index,
                    'task_type': type(sorted_tasks[task_index]).__name__,
                    'task_data': sorted_tasks[task_index].serialize()
                }
                self.comm.send(task_data, dest=worker_rank, tag=1)
                print(f"Enviada tarefa {task_index + 1} para processo {worker_rank}")
            
            # Processa tarefas e gerencia workers
            next_task_index = min(self.size - 1, total_tasks)
            
            while completed_tasks < total_tasks:
                # Recebe resultado de qualquer worker
                status = MPI.Status()
                result = self.comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
                worker_rank = status.Get_source()
                completed_tasks += 1
                
                print(f"Processo {worker_rank} concluiu: {result}")
                
                # Envia proxima tarefa se disponivel
                if next_task_index < total_tasks:
                    task_data = {
                        'task_index': next_task_index,
                        'task_type': type(sorted_tasks[next_task_index]).__name__,
                        'task_data': sorted_tasks[next_task_index].serialize()
                    }
                    self.comm.send(task_data, dest=worker_rank, tag=1)
                    print(f"Enviada tarefa {next_task_index + 1} para processo {worker_rank}")
                    next_task_index += 1
                else:
                    # Sinaliza para o worker parar
                    self.comm.send(None, dest=worker_rank, tag=1)
            
            parallel_time = time.time() - start_time
            print(f"\nTempo total paralelo: {parallel_time:.4f} segundos")
            return parallel_time
            
        else:
            # Processo worker
            while True:
                # Recebe tarefa do mestre
                task_data = self.comm.recv(source=0, tag=1)
                
                if task_data is None:
                    break  # Sinal para parar
                
                # Reconstroi a tarefa baseada no tipo
                task_type = task_data['task_type']
                task_obj = None
                
                if task_type == 'TrainingTask':
                    task_obj = TrainingTask.deserialize(task_data['task_data'])
                elif task_type == 'DataProcessingTask':
                    task_obj = DataProcessingTask.deserialize(task_data['task_data'])
                elif task_type == 'ModelEvaluationTask':
                    task_obj = ModelEvaluationTask.deserialize(task_data['task_data'])
                
                if task_obj:
                    # Executa a tarefa
                    result = task_obj.execute()
                    # Envia resultado de volta para o mestre
                    self.comm.send(result, dest=0, tag=2)
        
        return 0

def main():
    parser = argparse.ArgumentParser(description='Sistema de Treinamento Paralelo de IA')
    parser.add_argument('--modo', choices=['sequencial', 'paralelo', 'ambos'], 
                       default='ambos', help='Modo de execucao')
    args = parser.parse_args()
    
    scheduler = ParallelTrainingScheduler()
    
    if scheduler.rank == 0:
        print("SISTEMA DE TREINAMENTO DE IA - PROCESSAMENTO PARALELO")
        print(f"Processos MPI disponiveis: {scheduler.size}")
    
    # Cria lista de tarefas
    tasks = scheduler.create_tasks()
    
    # Executa conforme modo selecionado
    sequential_time = 0
    parallel_time = 0
    
    if args.modo in ['sequencial', 'ambos'] and scheduler.rank == 0:
        sequential_time = scheduler.sequential_execution(tasks.copy())
    
    # Sincroniza processos antes da execucao paralela
    scheduler.comm.Barrier()
    
    if args.modo in ['paralelo', 'ambos']:
        parallel_time = scheduler.parallel_execution(tasks)
    
    # Mostra comparacao de performance
    if scheduler.rank == 0 and args.modo == 'ambos' and sequential_time > 0 and parallel_time > 0:
        print("\n" + "="*50)
        print("COMPARACAO DE PERFORMANCE")
        print("="*50)
        speedup = sequential_time / parallel_time
        efficiency = (speedup / (scheduler.size - 1)) * 100
        
        print(f"Tempo Sequencial: {sequential_time:.4f}s")
        print(f"Tempo Paralelo: {parallel_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Eficiencia: {efficiency:.1f}%")
        print(f"Processos utilizados: {scheduler.size}")

if __name__ == "__main__":
    main()