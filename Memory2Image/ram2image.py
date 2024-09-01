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
import torchvision.models as models


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

class GameBoyDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM memory_data")
        self.length = self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.cursor.execute("SELECT ram_view, image FROM memory_data WHERE id=?", (idx+1,))
        ram_view_binary, image_binary = self.cursor.fetchone()

        # Process ram_view
        ram_view = np.load(io.BytesIO(ram_view_binary))
        ram_view = ram_view.astype(np.float32) / 255.0  # Normalize to [0, 1]
        ram_view = torch.from_numpy(ram_view)

        # Process image
        image = Image.open(io.BytesIO(image_binary))
        image = np.array(image.resize((160, 144)))[:,:,:3].transpose(2, 0, 1).astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        return ram_view, image

    def __del__(self):
        self.conn.close()

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class WRAMToImageModel(nn.Module):
    def __init__(self):
        super(WRAMToImageModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Reshape and apply convolutions
        self.reshaper = nn.Sequential(
            nn.Linear(1024, 256 * 4 * 5),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Upsampling
        self.upsample = nn.Upsample(size=(144, 160), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.reshaper(x)
        x = x.view(-1, 256, 4, 5)
        x = self.conv_layers(x)
        x = self.upsample(x)
        return x

# Perceptual loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:29].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target)
        return self.mse_loss(vgg_input, vgg_target)

# Training setup
db_path = "memory_data.db"
dataset = GameBoyDataset(db_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = WRAMToImageModel()
perceptual_loss = VGGPerceptualLoss()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 100
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
perceptual_loss.to(device)

epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for i, (ram_view, target_image) in enumerate(progress_bar):
        ram_view, target_image = ram_view.to(device), target_image.to(device)
        
        optimizer.zero_grad()
        output = model(ram_view)
        
        loss_mse = mse_loss(output, target_image)
        loss_perceptual = perceptual_loss(output, target_image)
        loss = loss_mse + 0.1 * loss_perceptual
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Save comparison image for the first batch of each epoch
        if i == 0:
            save_comparison_image(target_image[0], output[0], epoch+1)
    
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'gameboy_wram_to_screen_model_improved.pth')

# Plot loss rate
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), epoch_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.savefig("loss_plot_improved.png")
plt.close()

print("Training completed. Model saved as 'gameboy_wram_to_screen_model_improved.pth'")
print("Loss plot saved as 'loss_plot_improved.png'")