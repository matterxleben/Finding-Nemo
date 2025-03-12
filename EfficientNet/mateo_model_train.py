import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import signal
import sys
from torch.utils.data import DataLoader, random_split
from mateo_model import EfficientNetModel
from mateo_model_evaluation import evaluate_accuracy, mean_average_precision_at_20


# hyperparameters
batch_size = 32
lr = 1e-3
num_epochs = 200
threshold = 0.5
EVAL_EVERY = 10  # Evaluate every 10 epochs

# Evaluation results storage
eval_results = []
model = None  # Placeholder for model reference

def save_progress():
    if model is not None:
        torch.save(model.state_dict(), 'efficientnet_model_weights.pth')
        eval_df = pd.DataFrame(eval_results)
        eval_df.to_csv("evaluation_results.csv", index=False)
        print("Progress saved.")

# Handle keyboard interrupt (Ctrl+C) to save progress before exiting
def signal_handler(sig, frame):
    print("Training interrupted. Saving progress...")
    save_progress()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def train():
    global model
    # load in dataset and do test train split (80/20)
    dataset = torch.load('dataset.pt')
    torch.manual_seed(42)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Total samples in dataset: {total_samples}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize the model
    model = EfficientNetModel(num_classes=290)
    model = model.to("cpu")

    # define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch = 0
        for images, labels in train_loader:
            batch += 1
            print(f"Batch #{batch}")
            images, labels = images.to("cpu"), labels.to("cpu")

            # forward pass to get outputs and compute loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute total loss over number of samples in batch
            running_loss += loss.item() * images.size(0)

        # find average loss over the epoch    
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Evaluate every EVAL_EVERY epochs
        if (epoch + 1) % EVAL_EVERY == 0:
            print("Running Evaluation...")
            accuracy = evaluate_accuracy(model, test_loader, threshold)
            map20 = mean_average_precision_at_20(model, test_loader)
            eval_results.append({
                "epoch": epoch + 1,
                "accuracy": accuracy,
                "mAP@20": map20
            })
            print(f"Epoch {epoch+1}: Accuracy = {accuracy:.4f}, mAP@20 = {map20:.4f}")

    # Save final progress
    save_progress()

if __name__ == "__main__":
    train()
