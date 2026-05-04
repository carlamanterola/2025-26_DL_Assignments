import torch
import torch.nn as nn
import torch.optim as optim
import models.scratch as scratch
import models.vgg as vgg
import utils.plots as plots

# Set Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_engine(model, train_loader, val_loader, epochs, optimizer, criterion):
    """
    Unified training loop that tracks history for comparison plotting.
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        # Log metrics
        t_acc = 100. * train_correct / train_total
        v_acc = 100. * val_correct / val_total
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch}/{epochs} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")
        
    return history

def main():
    criterion = nn.CrossEntropyLoss()

    # 1. Train Scratch CNN (15 Epochs)
    print("--- Training Scratch CNN (32x32) ---")
    train_loader_s, val_loader_s = scratch.prepare_data()
    model_s = scratch.ScratchCNN(num_classes=10).to(DEVICE)
    optimizer_s = optim.SGD(model_s.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    history_scratch = train_engine(model_s, train_loader_s, val_loader_s, epochs=15, optimizer=optimizer_s, criterion=criterion)

    # 2. Train VGG Transfer Learning (5 Epochs)
    print("\n--- Training VGG Transfer Learning (224x224) ---")
    train_loader_v, val_loader_v = vgg.prepare_data()
    model_v = vgg.get_vgg_model(num_classes=10, freeze_features=True).to(DEVICE)
    optimizer_v = optim.SGD(model_v.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    history_vgg = train_engine(model_v, train_loader_v, val_loader_v, epochs=5, optimizer=optimizer_v, criterion=criterion)

    # 3. Generate Visualizations
    print("\nGenerating Comparison Plots...")
    plots.plot_training_curves(history_scratch, title='Scratch CNN Results', save_path='scratch_results.png')
    plots.plot_training_curves(history_vgg, title='VGG Transfer Learning Results', save_path='vgg_results.png')
    
    plots.plot_comparison(
        history_scratch, 
        history_vgg, 
        scratch_acc=history_scratch['val_acc'][-1], 
        vgg_acc=history_vgg['val_acc'][-1], 
        save_path='model_comparison.png'
    )
    print("Process complete. Comparison saved as 'model_comparison.png'.")

if __name__ == '__main__':
    main()