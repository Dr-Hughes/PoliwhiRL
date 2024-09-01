# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


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
            "SELECT mem_view, image FROM memory_data WHERE id=?", (idx + 1,)
        )
        mem_view_binary, image_binary = self.cursor.fetchone()

        # Process mem_view
        mem_view = np.load(io.BytesIO(mem_view_binary))
        mem_view = mem_view.reshape(18, 20).astype(np.float32)
        mem_view = torch.from_numpy(mem_view)

        # Process image
        image = Image.open(io.BytesIO(image_binary))
        image = (
            np.array(image.resize((160, 144)))[:, :, :3]
            .transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
        )
        image = torch.from_numpy(image)

        return mem_view, image

    def __del__(self):
        self.conn.close()


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 3, 1)

        # Initial upsampling to match U-Net input size
        self.initial_upsample = nn.Upsample(
            size=(72, 80), mode="bilinear", align_corners=True
        )

        # Final upsampling to match target image size
        self.final_upsample = nn.Upsample(
            size=(144, 160), mode="bilinear", align_corners=True
        )

    def forward(self, x):
        # Initial upsampling
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.initial_upsample(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.up_conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x, x1], dim=1))
        x = self.outc(x)

        # Final upsampling
        x = self.final_upsample(x)
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


# Training setup
db_path = "memory_data.db"
dataset = GameBoyDataset(db_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
device = torch.device("mps")
model.to(device)

epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, (mem_view, target_image) in enumerate(progress_bar):
        mem_view, target_image = mem_view.to(device), target_image.to(device)

        optimizer.zero_grad()
        output = model(mem_view)
        loss = criterion(output, target_image)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Save comparison image for the first batch of each epoch
        if i == 0:
            save_comparison_image(target_image[0], output[0], epoch + 1)

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "gameboy_screen_model.pth")

# Plot loss rate
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.savefig("loss_plot.png")
plt.close()

print("Training completed. Model saved as 'gameboy_screen_model.pth'")
print("Loss plot saved as 'loss_plot.png'")

# Example of how to use the trained model
# model.eval()
# test_mem_view = torch.randn(1, 18, 20).to(device)
# with torch.no_grad():
#     output_image = model(test_mem_view)
#     output_image = output_image.cpu().squeeze().permute(1, 2, 0).numpy()
#     output_image = (output_image * 255).astype(np.uint8)
#     Image.fromarray(output_image).save('output_image.png')
# print("Example output image saved as 'output_image.png'")
