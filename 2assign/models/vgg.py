import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models

# 1. Definición de la Arquitectura VGG (Transfer Learning)
def get_vgg_model(num_classes=10, freeze_features=True):
    # Cargamos VGG16 pre-entrenado
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Congelamos las capas convolucionales iniciales
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Reemplazamos el clasificador para adaptarlo a nuestras clases
    # El valor 25088 es el resultado de la salida de VGG features (512*7*7)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),
    )
    return model

# 2. Configuración de Dispositivo y Datos
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

def prepare_data():
    # VGG requiere imágenes de 224x224 para funcionar con sus pesos originales
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# 3. Lógica de Entrenamiento (Siguiendo tu ejemplo de Scratch)
def train_vgg():
    train_loader, val_loader = prepare_data()
    
    # Inicializamos VGG con las capas base congeladas
    model = get_vgg_model(num_classes=10, freeze_features=True).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    epochs = 5  # El Transfer Learning suele requerir menos épocas
    print(f"Entrenando VGG en: {DEVICE}")

    for epoch in range(1, epochs + 1):
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

    torch.save(model.state_dict(), 'vgg_transfer_v1.pth')
    print("Modelo guardado como vgg_transfer_v1.pth")

if __name__ == '__main__':
    train_vgg()