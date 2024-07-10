import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, self.data.columns.get_loc('fname')]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.data.iloc[idx, self.data.columns.get_loc('gt_label')], dtype=torch.float32)
        return image, label

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, self.data.columns.get_loc('fname')]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

def get_data_loaders(train_csv, test_csv, batch_size=32):
    # Load train dataset
    train_data = pd.read_csv(train_csv)

    # Split train dataset into train and validation sets
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create dataset objects
    train_dataset = TrainDataset(train_data)
    val_dataset = TrainDataset(val_data)
    test_data = pd.read_csv(test_csv)
    test_dataset = TestDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained ResNet-50 model
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification output
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)  # Ensure labels have the correct shape

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)  # Ensure labels have the correct shape
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    return val_accuracy

def infer_and_save_results(model, test_loader, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    results = []

    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            predicted = torch.sigmoid(outputs).round().cpu().numpy().astype(int)
            results.extend(predicted.flatten())
    print("this is the results array", results)
    # Load test data to save results
    test_data = test_loader.dataset.data
    test_data['label'] = results

    out_dir = os.path.dirname(output_csv)
    os.makedirs(out_dir, exist_ok=True)
    test_data.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')

def main():
    in_csv = os.path.join(os.path.dirname(__file__), '..', 'input_data', 'test', 'test.csv')
    out_csv = os.path.join(os.path.dirname(__file__), '..', 'output_data', 'submission.csv')
    train_csv = os.path.join(os.path.dirname(__file__), '..', 'input_data', 'train', 'train.csv')

    train_loader, val_loader, test_loader = get_data_loaders(train_csv, in_csv, batch_size=32)
    model = train_model(train_loader, val_loader, num_epochs=30)
    infer_and_save_results(model, test_loader, out_csv)

    pass


# Example usage
if __name__ == "__main__":
    main()