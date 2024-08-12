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
        self.enc1 = self._make_encoder_layer(input_channels, 64)
        self.enc2 = self._make_encoder_layer(64, 128)
        self.enc3 = self._make_encoder_layer(128, 256)
        self.enc4 = self._make_encoder_layer(256, 512)

        # Decoder
        self.dec1 = self._make_decoder_layer(512, 256)
        self.dec2 = self._make_decoder_layer(256, 128)
        self.dec3 = self._make_decoder_layer(128, 64)
        self.dec4 = self._make_decoder_layer(64, 32)

        # Final convolution
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def _make_encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
        )

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoding with skip connections
        d1 = self.dec1(e4)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)

        # Final convolution
        x = self.final_conv(d4)

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

# Load the model if it exists
if os.path.exists("memory_to_image_model.pth"):
    print("Loading model from memory_to_image_model.pth")
    model.load_state_dict(torch.load("memory_to_image_model.pth"))

# Load dataset
dataset = MemoryDataset("../RandomExplorer/memory_data.db")

# Hyperparameters
batch_size = 64
learning_rate = 0.0001
num_epochs = 1000

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

    print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "memory_to_image_model.pth")
