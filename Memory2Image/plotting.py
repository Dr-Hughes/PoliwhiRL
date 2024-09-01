# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt


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


def plot_loss(num_epochs, epoch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.savefig("loss_plot.png")
    plt.close()
