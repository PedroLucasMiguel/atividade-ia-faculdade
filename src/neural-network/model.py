import torch
from torch import nn

class NetworkInputSizeInvalidException(Exception):
    "The input size is not valid! We expect a tuple like this: (int, int)"
    pass

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize:tuple) -> None:
        super(NeuralNetwork, self).__init__()

        if len(inputSize) != 2:
            raise NetworkInputSizeInvalidException

        if (not isinstance(inputSize[0], int)) or (not isinstance(inputSize[1], int)):
            raise NetworkInputSizeInvalidException

        self.flatten = nn.Flatten() # Isso converte o vetor de entrada em um vetor de uma só dimensão
        self.sequential = nn.Sequential(
            nn.Linear(inputSize[0] * inputSize[1], 16), # Camada de entrada
            nn.ReLU(), # Função de ativação ReLU (usada semrpe na saída das demais)
            nn.Linear(16, 16), # Primeira Hidden Layer
            nn.ReLU(),
            nn.Linear(16, 16), # Segunda Hidden Layer
            nn.ReLU(),
            nn.Linear(16, 10) # Camada de saída
        )

    def forward(self, x):
        return self.sequential(self.flatten(x))
