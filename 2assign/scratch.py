import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import copy

# 1. Definición de la Arquitectura (Estilo AlexNet simplificado)
class ScratchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ScratchCNN, self).__init__()
        self.features = nn.Sequential(
            # Capa 1: Entrada 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Capa 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Capas profundas
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Salida 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. Configuración de Dispositivo y Datos
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

def prepare_data():
    # Usamos los transforms de 'scratch' (32x32) de tu código original
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# 3. Lógica de Entrenamiento (Basada en tu train_one_epoch y evaluate)
def train_model():
    train_loader, val_loader = prepare_data()
    model = ScratchCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    epochs = 15
    print(f"Entrenando en: {DEVICE}")

    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100. * correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss/total:.4f} | Acc: {train_acc:.2f}%")

    torch.save(model.state_dict(), 'scratch_cnn_v1.pth')
    print("Modelo guardado como scratch_cnn_v1.pth")

if __name__ == '__main__':
    train_model()