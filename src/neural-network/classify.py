import os
import torch
import torchvision.transforms as transforms
import torchvision
from model import NeuralNetwork

# Transformação padrão para todos o dataset
transform = transforms.Compose([transforms.ToTensor()])

# Criando dataset de validação (10000 amostras)
# Este dataset é utilizado neste arquivo pois o modelo não foi treinado
# nestas imagens
validation_set = torchvision.datasets.MNIST(
    root=os.path.join("."), 
    train=False,
    transform=transform, 
    download=False
    )

# Loader com batch size 1 para que todas as imagens sejam percorridas em um loop
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1)

#Utiliza GPU caso possível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# Cria o modelo
model = NeuralNetwork(validation_set.data[0].shape)

# Carrega o melhor treinamento
model.load_state_dict(torch.load(os.path.join("model_checkpoint", "best_model_45_f1=0.9283.pt")))

model = model.to(device)

# Habilitando modo de classificação do modelo
model.eval()

# Percorre o dataset realizando as classificações
for digit, label in validation_loader:
    pred = model(digit.to(device))
    pred_value, pred_label = torch.max(pred, 1)
    print(f"Label: {label[0]} | Prediction: {pred_label[0]} | Prediction value: {pred_value[0]}")