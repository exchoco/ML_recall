import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr

        # Contracting path
        self.enc1 = nn.Sequential(
            CBR2d(in_channels=in_channels, out_channels=64),
            CBR2d(in_channels=64, out_channels=64)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = nn.Sequential(
            CBR2d(in_channels=64, out_channels=128),
            CBR2d(in_channels=128, out_channels=128)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = nn.Sequential(
            CBR2d(in_channels=128, out_channels=256),
            CBR2d(in_channels=256, out_channels=256)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc4 = nn.Sequential(
            CBR2d(in_channels=256, out_channels=512),
            CBR2d(in_channels=512, out_channels=512)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.enc5 = nn.Sequential(
            CBR2d(in_channels=512, out_channels=1024),
            CBR2d(in_channels=1024, out_channels=1024)
        )

        # Expansive path
        self.up4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4 = nn.Sequential(
            CBR2d(in_channels=1024, out_channels=512),
            CBR2d(in_channels=512, out_channels=512)
        )
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3 = nn.Sequential(
            CBR2d(in_channels=512, out_channels=256),
            CBR2d(in_channels=256, out_channels=256)
        )
        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2 = nn.Sequential(
            CBR2d(in_channels=256, out_channels=128),
            CBR2d(in_channels=128, out_channels=128)
        )
        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1 = nn.Sequential(
            CBR2d(in_channels=128, out_channels=64),
            CBR2d(in_channels=64, out_channels=64)
        )
        self.out_layer = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc5 = self.enc5(self.pool4(enc4))

        dec4 = self.dec4(torch.cat((self.up4(enc5), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))

        x = self.out_layer(dec1)

        return x

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(os.path.dirname(__file__),self.data.iloc[idx]['fname'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        out_img_path = os.path.join(os.path.dirname(__file__),self.data.iloc[idx]['gt_out_fname'])
        output_image = Image.open(out_img_path).convert('RGB')
        output_image = self.transform(output_image)

        return image, output_image

class TestDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(os.path.dirname(__file__),self.data.iloc[idx]['fname'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path

def get_data_loaders(train_csv, test_csv, batch_size=32):
    # Load train dataset
    train_data = pd.read_csv(train_csv)

    # Split train dataset into train and validation sets
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create dataset objects
    train_dataset = TrainDataset(train_data)
    val_dataset = TrainDataset(val_data)
    test_dataset = TestDataset(test_csv)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for input_image, target_image in train_loader:
            input_image, target_image = input_image.to(device), target_image.to(device)

            optimizer.zero_grad()

            output_image = model(input_image)
            loss = criterion(output_image, target_image)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for input_image, target_image in val_loader:
                input_image, target_image = input_image.to(device), target_image.to(device)

                output_image = model(input_image)
                loss = criterion(output_image, target_image)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    return model

def infer_and_save_results(model, test_loader, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    results = []

    with torch.no_grad():
        for images, img_paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            output_images = outputs.cpu()

            for img, path in zip(output_images, img_paths):
                result_path = path.replace('.png', '_out.png')
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(result_path)
                results.append(result_path)

    # Load test data to save results
    test_data = test_loader.dataset.data
    test_data['out_fname'] = results

    out_dir = os.path.dirname(output_csv)
    os.makedirs(out_dir, exist_ok=True)
    test_data.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')

def main():
    train_csv = os.path.join(os.path.dirname(__file__), '..', 'input_data', 'train', 'train.csv')
    test_csv = os.path.join(os.path.dirname(__file__), '..', 'input_data', 'test', 'test.csv')
    output_csv = os.path.join(os.path.dirname(__file__), '..', 'output_data', 'submission.csv')

    train_loader, val_loader, test_loader = get_data_loaders(train_csv, test_csv, batch_size=8)
    model = UNet(in_channels=3, out_channels=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50)
    infer_and_save_results(model, test_loader, output_csv)



if __name__ == '__main__':
    main()