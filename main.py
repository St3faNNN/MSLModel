import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from get_model import get_model
import numpy as np

def train_model(model, loader, criterion, optimizer, epoch, model_name, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/15 [{model_name}]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss, acc

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

if __name__ == "__main__":
    data = r'C:\Users\stefa\Desktop\MSLModel\mixed_dataset'
    batch_size = 32
    learning_rate = 0.001
    train_split = 0.8
    seed = 42
    patience = 3
    model_names = ['mobilenet', 'resnet', 'efficientnet']

    torch.manual_seed(seed)


    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


    full_dataset = datasets.ImageFolder(data)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    targets = np.array([s[1] for s in full_dataset.samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)


    best_model_name = ""
    best_val_acc = 0.0

    for model_name in model_names:
        print(f"\nTraining model: {model_name}")
        model = get_model(model_name, num_classes).to("cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        best_epoch = 0
        early_stop = False

        for epoch in range(15):
            if early_stop:
                print(f"Rano zapiranje {epoch+1}")
                break

            loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch, model_name, "cpu")
            print(f"epoha {epoch+1}: Loss = {loss:.4f} | Train accuracy {train_acc:.2f}%")

            val_acc = evaluate_model(model, val_loader, "cpu")
            print(f"Validation Accuracy: {val_acc:.2f}%")

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_name = model_name
                best_epoch = epoch
                torch.save(model.state_dict(), f"best_model_{model_name}.pth")
            elif epoch - best_epoch >= patience:
                early_stop = True

    print(f"\nNajdobar model: {best_model_name}")
