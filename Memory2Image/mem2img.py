# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("mps")


class MemoryDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM memory_data")
        return self.cursor.fetchone()[0]

    def __getitem__(self, index):
        self.cursor.execute(
            "SELECT mem_view, image FROM memory_data WHERE id=?", (index + 1,)
        )
        row = self.cursor.fetchone()
        mem_view_str = row[0]
        img_bytes = row[1]
        mem_view = np.array(eval(mem_view_str))
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)
        mem_view_tensor = torch.from_numpy(mem_view).float()
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        return mem_view_tensor, img_tensor

    def __del__(self):
        self.conn.close()


class MemoryToImageCNN(nn.Module):
    def __init__(self, input_channels=1, output_height=144, output_width=160):
        super(MemoryToImageCNN, self).__init__()
        self.output_height = output_height
        self.output_width = output_width

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        # Final convolution
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoding
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        # Final convolution
        x = self.final_conv(x)

        # Resize to the exact output dimensions
        x = F.interpolate(
            x,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=False,
        )

        # Apply sigmoid to ensure output is between 0 and 1
        x = torch.sigmoid(x)
        return x


def save_comparison_image(original, generated, epoch, output_folder="mem2img", i=0):
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


# Create the model and move it to the device
model = MemoryToImageCNN(input_channels=1, output_height=144, output_width=160).to(
    device
)

# Print model summary
print(model)

# Load dataset
dataset = MemoryDataset("memory_data.db")

# Hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 100

# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (batch_mem_views, batch_images) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        # Move data to device
        batch_mem_views = batch_mem_views.unsqueeze(1).to(device)
        batch_images = batch_images.permute(0, 3, 1, 2).to(device)

        # Forward pass
        outputs = model(batch_mem_views)

        # Ensure batch_images matches the model output size
        batch_images = F.interpolate(
            batch_images, size=(144, 160), mode="bilinear", align_corners=False
        )

        # Compute loss
        loss = criterion(outputs, batch_images)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save comparison image for the first batch of each epoch
        if i == 0:
            for j in range(min(batch_images.size(0), 5)):
                save_comparison_image(batch_images[j], outputs[j], epoch, i=j)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "memory_to_image_model.pth")
