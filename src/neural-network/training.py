import os
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from model import NeuralNetwork
from ignite.engine import Events
from ignite.metrics import *
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
import json

metrics_json = {}

# Transformação padrão para todos os datasets
transform = transforms.Compose([transforms.ToTensor()])

# Criando dataset de treino (60000 amostras)
# O Dataset será baixado caso não exista
training_set = torchvision.datasets.MNIST(
    root=os.path.join("."), 
    train=True,
    transform=transform, 
    download=True
    )

# Criando dataset de validação (10000 amostras)
validation_set = torchvision.datasets.MNIST(
    root=os.path.join("."), 
    train=False,
    transform=transform, 
    download=False
    )

# Criando o loader do dataset de treino
train_loader = torch.utils.data.DataLoader(training_set, batch_size=256)

# Criando o loado do dataset de validação
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=256)

#Utiliza GPU caso possível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# Cria o modelo
model = NeuralNetwork(training_set.data[0].shape).to(device)

# Definindo criterion e optmizer
# https://notebook.community/jmhsi/justin_tinker/data_science/courses/temp/courses/ml1/lesson4-mnist_sgd
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.003, lr=0.1)

# Realizando o processo de treinamento

# Define o processo de treinamento da rede
def train_step(engine, batch):
    # x = imagem, y = label
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    # Habilita o modo de treino do modelo
    model.train()
    # Realiza a predição
    y_pred = model(x)
    # Calcula o loss
    loss = criterion(y_pred, y)
    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# Define o processo de validação da rede
def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        # x = imagem, y = label
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # Realiza a predição
        y_pred = model(x)
        
        # Retorna a saída da rede com a classe esperada
        return y_pred, y

# Cria o treiner
trainer = Engine(train_step)

# Cria o evaluator
evaluator = Engine(validation_step)

# Cálculo da medida-F
def F1():
    precision = Precision(average=False)
    recall = Recall(average=False)
    return (precision * recall * 2 / (precision + recall)).mean()

# Dicionário com as métricas que queremos obter
val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion),
    "precision": Precision(average=True),
    "recall": Recall(average=True),
    "f1": F1()
}

# Adiciona as métricas ao trainer e ao evaluator
for name, metric in val_metrics.items():
    metric.attach(evaluator, name)
    
# Evento no inicio do treino
@trainer.on(Events.STARTED)
def startMessage():
    print("Start training!")

# Evento de fim de treino
@trainer.on(Events.COMPLETED)
def endMessage():
    print("Training completed!")

# Após cada epoch, realiza um processo de validação
@trainer.on(Events.EPOCH_COMPLETED)
def runValidation():
    evaluator.run(validation_loader)

# Mostra as métricas após cada processo de validação
@evaluator.on(Events.COMPLETED)
def log_validation_results():
    metrics = evaluator.state.metrics
    print("\n\nValidation Results - Epoch: {}\nAccuracy: {:.3f}\nLoss: {:.3f}\nF1: {:.3f}\nPrecision: {:.3f}\nRecall {:.3f}\n\n"
        .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"], metrics["f1"], metrics["precision"], metrics["recall"]))
    
    metrics_json[trainer.state.epoch] = {
        "Accuracy": metrics["accuracy"],
        "Loss": metrics["loss"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
    }

# Adiciona uma barra de progresso apenas para melhor visualizar como o treinamento está indo :^)
ProgressBar().attach(trainer, output_transform=lambda x: {'batch loss': x})

# Define qual será a métrica que definira um checkpoint
def score_function(engine):
    return engine.state.metrics["f1"]

# Checkpoints!
# Salva o treinamento da rede baseado no resultado da medida F1
# Essa etapa previne que salvemos treinamentos que estão apontando overfitting :^D
model_checkpoint = ModelCheckpoint(
    "model_checkpoint",
    filename_prefix="best",
    score_function=score_function,
    score_name="f1",
    require_empty=False,
    global_step_transform=global_step_from_engine(trainer),
)

# Se o modelo for relevante, salvamos o treinamento deste modelo após cada etapa de validação
evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

# Começa o treinamento em si \O/
trainer.run(train_loader, max_epochs=50)

with open(os.path.join(".", "results.json"), "w") as json_file:
    json.dump(metrics_json, json_file)

