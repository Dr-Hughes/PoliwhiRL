# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import sqlite3
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F
from plotting import save_comparison_image


class GameBoyDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM memory_data")
        self.length = self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.cursor.execute(
            "SELECT ram_view, image FROM memory_data WHERE id=?", (idx + 1,)
        )
        ram_view_binary, image_binary = self.cursor.fetchone()

        # Process ram_view
        ram_view = np.load(io.BytesIO(ram_view_binary))
        ram_view = ram_view.astype(np.float32)
        ram_view = (ram_view - ram_view.mean()) / ram_view.std()  # Standardization
        ram_view = torch.from_numpy(ram_view)

        # Process image
        image = Image.open(io.BytesIO(image_binary))
        image = np.array(image)[:, :, :3].transpose(2, 0, 1).astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        return ram_view, image

    def __del__(self):
        self.conn.close()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class WRAMToImageModel(nn.Module):
    def __init__(self):
        super(WRAMToImageModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Latent space
        self.latent = nn.Sequential(
            nn.Linear(512 * 1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 18 * 20 * 64),  # Adjusted to match 144x160 output
        )

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(64, 128),
            nn.Upsample(scale_factor=2),  # 36x40
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2),  # 72x80
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2),  # 144x160
            ConvBlock(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Input shape: (batch_size, 8192)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, 8192)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.latent(x)
        x = x.view(x.size(0), 64, 18, 20)  # Adjusted to 18x20
        x = self.decoder(x)
        return x  # Output shape: (batch_size, 3, 144, 160)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        h_x = x
        h_target = target
        h1_x = self.slice1(h_x)
        h1_target = self.slice1(h_target)
        h2_x = self.slice2(h1_x)
        h2_target = self.slice2(h1_target)
        loss = F.mse_loss(h1_x, h1_target) + F.mse_loss(h2_x, h2_target)
        return loss


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (ram_view, target_image) in enumerate(progress_bar):
            ram_view, target_image = ram_view.to(device), target_image.to(device)

            optimizer.zero_grad()
            output = model(ram_view)
            loss = criterion(output, target_image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Save comparison image for the first batch of each epoch
            if i == 0:
                save_comparison_image(target_image[0], output[0], epoch + 1)

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ram_view, target_image in val_loader:
                ram_view, target_image = ram_view.to(device), target_image.to(device)
                output = model(ram_view)
                loss = criterion(output, target_image)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

    return model


# Main execution
if __name__ == "__main__":
    db_path = "memory_data.db"
    dataset = GameBoyDataset(db_path)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = WRAMToImageModel()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    criterion = nn.MSELoss()
    perceptual_criterion = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )

    # Test the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for ram_view, target_image in test_loader:
            ram_view, target_image = ram_view.to(device), target_image.to(device)
            output = model(ram_view)
            loss = criterion(output, target_image)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    print("Training and evaluation completed.")
