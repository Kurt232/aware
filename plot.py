import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot(output_dir):
    log_path = os.path.join(output_dir, 'log.txt')
    data = open(log_path, 'r').read()
    # Parse the data into a list of dictionaries
    data_list = []
    for line in data.strip().split('\n'):
        if not line:
            continue
        data_list.append(json.loads(line))

    # Convert the list into a pandas DataFrame
    df = pd.DataFrame(data_list)

    # Plotting
    if 'train_acc' not in df.columns:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Add a title for the entire plot
    fig.suptitle(f"Training Progress: {os.path.basename(output_dir)}")
    # Subplot 1: Learning Rate vs. Epoch
    axs[0].plot(df['epoch'], df['train_lr'], color='blue')
    axs[0].set_title('Learning Rate vs. Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Learning Rate')
    axs[0].grid(True)

    # Subplot 2: Losses vs. Epoch
    axs[1].plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        axs[1].plot(df['epoch'], df['val_loss'], label='Validation Loss')
    axs[1].set_title('Loss vs. Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    if 'train_acc' in df.columns:
        # Subplot 3: Accuracies vs. Epoch
        axs[2].plot(df['epoch'], df['train_acc'], label='Train Accuracy')
        if 'val_acc' in df.columns:
            axs[2].plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
        axs[2].set_title('Accuracy vs. Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Accuracy (%)')
        axs[2].set_ylim([0, 100])
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    root = sys.argv[1]
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for d in dirs:
        path = os.path.join(root, d)
        print(path)
        plot(path)