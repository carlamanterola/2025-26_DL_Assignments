import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. Cargamos el dataset desde Hugging Face (Mucho más estable)
print("Cargando CIFAR-10 desde Hugging Face...")
raw_datasets = load_dataset("cifar10")

# 2. Definimos las transformaciones (Normalización estándar de CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 3. Función para preparar los datos
def transform_fn(examples):
    # Convertimos las imágenes a tensores y aplicamos normalización
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["img"]]
    return examples

# Aplicamos la transformación al vuelo
transformed_dataset = raw_datasets.with_transform(transform_fn)

# 4. Función para agrupar los datos en batches (Collate function)
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return pixel_values, labels

# 5. Creamos los DataLoaders
trainloader = DataLoader(
    transformed_dataset["train"], 
    batch_size=64, 
    shuffle=True, 
    collate_fn=collate_fn
)

testloader = DataLoader(
    transformed_dataset["test"], 
    batch_size=64, 
    shuffle=False, 
    collate_fn=collate_fn
)

# --- Verificación ---
images, labels = next(iter(trainloader))
print(f"✅ ¡Dataset listo!")
print(f"Estructura del batch: {images.shape}") # Debería ser [64, 3, 32, 32]
print(f"Etiquetas del batch: {labels}")