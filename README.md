# Programacao_Paralela_Sistemas
Sistema de demonstração que simula treinamento de modelos de IA usando processamento paralelo com MPI. Desenvolvido para ilustrar conceitos de paralelismo e escalonamento de prioridades.

## Execução Sequencial
python main.py --modo sequencial

## Execução Paralela
mpiexec -n 4 python main.py --modo paralelo

## Execução Comparativa
mpiexec -n 4 python main.py --modo ambos

### Pré-requisitos
```bash
pip install mpi4py numpy
pip install -r requirements.txt

