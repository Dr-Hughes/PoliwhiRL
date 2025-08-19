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
import os
import matplotlib.pyplot as plt


def save_comparison_image(
    original, generated, epoch, output_folder="action2ram2image", i=0
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Convert tensors to numpy arrays and transpose to (H, W, C)
    original = original.cpu().numpy().transpose(1, 2, 0)
    generated = generated.cpu().detach().numpy().transpose(1, 2, 0)
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Plot original image
    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.axis("off")
    # Plot generated image
    ax2.imshow(generated)
    ax2.set_title("Generated")
    ax2.axis("off")
    # Save the figure
    plt.savefig(os.path.join(output_folder, f"comparison_epoch_{epoch}_{i}.png"))
    plt.close(fig)


class GameBoyDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM memory_data")
        self.length = self.cursor.fetchone()[0]
        self.episode_boundaries = self._get_episode_boundaries()

    def _get_episode_boundaries(self):
        self.cursor.execute("SELECT id, episode_id FROM memory_data ORDER BY id")
        data = self.cursor.fetchall()
        boundaries = []
        current_episode = data[0][1]
        start = 0
        for i, (_, episode_id) in enumerate(data):
            if episode_id != current_episode:
                boundaries.append((start, i - 1))
                start = i
                current_episode = episode_id
        boundaries.append((start, len(data) - 1))
        return boundaries

    def __len__(self):
        return sum(end - start for start, end in self.episode_boundaries) - len(
            self.episode_boundaries
        )

    def __getitem__(self, idx):
        for start, end in self.episode_boundaries:
            if idx < end - start - 1:  # Subtract 1 to ensure we have a next state
                real_idx = start + idx
                break
            idx -= end - start - 1

        # Fetch current state
        self.cursor.execute(
            "SELECT ram_view, image FROM memory_data WHERE id=?", (real_idx + 1,)
        )
        current_ram, current_image = self.cursor.fetchone()

        # Fetch next state and action
        self.cursor.execute(
            "SELECT ram_view, image, action FROM memory_data WHERE id=?",
            (real_idx + 2,),
        )
        next_ram, next_image, action = self.cursor.fetchone()

        # Process ram_views
        current_ram = self._process_ram(current_ram)
        next_ram = self._process_ram(next_ram)

        # Process images
        current_image = self._process_image(current_image)
        next_image = self._process_image(next_image)

        # Process action
        action = torch.tensor(action, dtype=torch.long)

        return current_ram, action, next_ram, current_image, next_image

    def _process_ram(self, ram_binary):
        ram = np.load(io.BytesIO(ram_binary))
        ram = ram.astype(np.float32)
        ram = (ram - ram.mean()) / ram.std()  # Standardization
        return torch.from_numpy(ram)

    def _process_image(self, image_binary):
        image = Image.open(io.BytesIO(image_binary))
        image = np.array(image)[:, :, :3].transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(image)

    def __del__(self):
        self.conn.close()


class StateTransitionModel(nn.Module):
    def __init__(self, input_size=8192, hidden_size=1024, action_size=9):
        super(StateTransitionModel, self).__init__()
        self.fc1 = nn.Linear(input_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def forward(self, ram, action):
        action_one_hot = F.one_hot(action, num_classes=9).float()
        x = torch.cat([ram, action_one_hot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ImagePredictionModel(nn.Module):
    def __init__(self):
        super(ImagePredictionModel, self).__init__()
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
        self.latent = nn.Sequential(
            nn.Linear(512 * 1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 18 * 20 * 64),
        )
        self.decoder = nn.Sequential(
            ConvBlock(64, 128),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2),
            ConvBlock(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.latent(x)
        x = x.view(x.size(0), 64, 18, 20)
        x = self.decoder(x)
        return x


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


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.state_transition = StateTransitionModel()
        self.image_prediction = ImagePredictionModel()

    def forward(self, ram, action):
        next_ram = self.state_transition(ram, action)
        next_image = self.image_prediction(next_ram)
        return next_ram, next_image


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device):
    ram_criterion = nn.MSELoss()
    image_criterion = nn.MSELoss()
    perceptual_criterion = PerceptualLoss().to(device)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (current_ram, action, next_ram, current_image, next_image) in enumerate(
            progress_bar
        ):
            current_ram, action, next_ram, current_image, next_image = (
                current_ram.to(device),
                action.to(device),
                next_ram.to(device),
                current_image.to(device),
                next_image.to(device),
            )

            optimizer.zero_grad()
            predicted_next_ram, predicted_next_image = model(current_ram, action)

            ram_loss = ram_criterion(predicted_next_ram, next_ram)
            image_loss = image_criterion(predicted_next_image, next_image)
            perceptual_loss = perceptual_criterion(predicted_next_image, next_image)

            loss = ram_loss + image_loss + 0.1 * perceptual_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if i % 20 == 0:
                save_comparison_image(
                    next_image[0], predicted_next_image[0], epoch + 1, i=str(i)
                )

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for current_ram, action, next_ram, current_image, next_image in val_loader:
                current_ram, action, next_ram, current_image, next_image = (
                    current_ram.to(device),
                    action.to(device),
                    next_ram.to(device),
                    current_image.to(device),
                    next_image.to(device),
                )
                predicted_next_ram, predicted_next_image = model(current_ram, action)
                ram_loss = ram_criterion(predicted_next_ram, next_ram)
                image_loss = image_criterion(predicted_next_image, next_image)
                perceptual_loss = perceptual_criterion(predicted_next_image, next_image)
                loss = ram_loss + image_loss + 0.1 * perceptual_loss
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

    model = CombinedModel()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    model = train_model(model, train_loader, val_loader, optimizer, num_epochs, device)

    # Test the model
    model.eval()
    test_loss = 0
    ram_criterion = nn.MSELoss()
    image_criterion = nn.MSELoss()
    perceptual_criterion = PerceptualLoss().to(device)

    with torch.no_grad():
        for current_ram, action, next_ram, current_image, next_image in test_loader:
            current_ram, action, next_ram, current_image, next_image = (
                current_ram.to(device),
                action.to(device),
                next_ram.to(device),
                current_image.to(device),
                next_image.to(device),
            )
            predicted_next_ram, predicted_next_image = model(current_ram, action)
            ram_loss = ram_criterion(predicted_next_ram, next_ram)
            image_loss = image_criterion(predicted_next_image, next_image)
            perceptual_loss = perceptual_criterion(predicted_next_image, next_image)
            loss = ram_loss + image_loss + 0.1 * perceptual_loss
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    print("Training and evaluation completed.")
