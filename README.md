# Trabalho prático de Inteligência Artificial (2023)
Repositório dedicado a documentação/organização da atividade de Inteligência Artificial.

Este projeto teve como intuito o uso de três algoritmos de classificação para avaliar o ["MNIST Dataset"] (https://en.wikipedia.org/wiki/MNIST_database).

Os algoritmos selecionados para realizar a classificação foram:
- `Decision tree`;
  - Scikit-learn 
- `Support Vector Machine (SVM)`;
  - Scikit-learn 
- `Neural Network (3 hidden layers)`;
  - Pytorch 

## Estrutura do projeto
O projeto está organizado da seguinte forma:
- `src/` -> Diretório que contém todos os códigos responsáveis pela execução dos três algoritmos selecionados;
  - `common/` -> Utilitários para os modelos do sckit-learn;
  - `decision-tree/` -> Código para o classificador de árvore de decisão;
  - `neural-network/` -> Código para o modelo de rede neural com três "hidden layers";
  - `support-vector-machine/` -> Código para o classificador SVM;

- `generated/` -> Diretório que contém todas as "saídas" dos classificadores;
  - `models/` -> Serialização dos treinamentos dos modelos;
  - `results/` -> Serialização das métricas adquiridas durante os treinamentos dos classificadores;

## Especificações do projeto

- Ambiente Python em sua versão 3.10 (versões diferentes à indicada não foram testadas, e podem causar instabilidades);
- `Pytorch e Pytorch-ignite`
- `Scikit-learn`

### Criando ambiente para execução

Em uma instância do Python em sua versão 3.10, na raiz do projeto, execute os seguintes comandos:

- Instalação Scikit-learn:
```shell
pip install -r scikit-requirements.txt
````

- Instalação do Pytorch:
  - Para realizar a instalção do Pytorch é recomendado que siga as instruções fornecidas [aqui.](https://pytorch.org/get-started/locally/)

- Instalação do Pytorch-ignite:
```shell
pip install pytorch-ignite
```

## Executando o projeto

Para executar cada um dos classificadores, primeiro certifique-se que você está na pasta do respectivo classificador:

- `src/decision-tree/` -> Árvore de decisão
- `src/neural-network/` -> Neural Network
- `src/support-vector-machine/` -> SVM

- Executar o classificador "decision-tree" (resultados exportados para `generated/results/decicion-tree.txt`):
```shell
python main.py
```

- Treinar a "neural-network" (resultados exportados para `generated/results/neural-network.json`):
```shell
python training.py
```

- Classificar com a "neural-network" (resultados no terminal):
```shell
python classify.py
```

- Executar o classificador "support-vector-machine/" (resultados exportados para `generated/results/"support-vector-machine.txt`);
```shell
python main.py
```


    

