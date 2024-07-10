import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust size as needed
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, self.data.columns.get_loc('fname')])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        out_img_name = os.path.join(self.root_dir, self.data.iloc[idx, self.data.columns.get_loc('gt_out_fname')])
        output_image = Image.open(out_img_name).convert('RGB')
        output_image = self.transform(output_image)

        return image, output_image

class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust size as needed
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, self.data.columns.get_loc('fname')])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        return image, img_name  # Return image and input image path for inference

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define your generator architecture here
        # Example: U-Net generator
        self.encoder = nn.Sequential(
            # Encoder layers
        )
        self.decoder = nn.Sequential(
            # Decoder layers
        )

    def forward(self, x):
        # Forward pass through the generator
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for input_image, target_image in train_loader:
            input_image, target_image = input_image.to(device), target_image.to(device)

            # Forward pass
            output_image = model(input_image)

            # Compute loss
            loss = criterion(output_image, target_image)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def infer_and_save_results(model, test_loader):
    model.eval()
    with torch.no_grad():
        for input_image, img_path in test_loader:
            input_image = input_image.to(device)
            output_image = model(input_image)
            output_image = transforms.ToPILImage()(output_image[0].cpu())  # Assuming batch size 1
            out_path = os.path.splitext(img_path)[0] + '_out.png'
            output_image.save(out_path)
            test_data.at[test_data['fname'] == img_path, 'gt_out_fname'] = out_path
    test_data.to_csv('submission.csv', index=False)
    print('Results saved to submission.csv')


def main():
    train_csv = 'path/to/train.csv'
    test_csv = 'path/to/test.csv'
    output_csv = 'submission.csv'

    train_dataset = TrainDataset(train_csv, root_dir='path/to/train/images')
    test_dataset = TestDataset(test_csv, root_dir='path/to/test/images')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Generator().to(device)
    criterion = nn.MSELoss()  # Example loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, criterion, optimizer, num_epochs=10)
    infer_and_save_results(model, test_loader)

if __name__ == '__main__':
    main()