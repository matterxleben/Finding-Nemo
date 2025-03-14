import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pandas as pd
import numpy as np
import sys
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from image_dataset import MultiLabelImageDataset

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 3

# File Paths
LABELS_PATH = "data/train_classification_labels.csv"
IMAGES_PATH = "data/train"
MODEL_SAVE_PATH = "densenet_model.pth"
CSV_LOG_PATH = "densenet_training_log.csv"

#Using densenet121 adapted for multi-label classification

#Use default function params
def create_densenet(num_classes=290, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)

    in_features = model.classifier.in_features
    
    # replace with a linear layer for 290 classes
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def mean_average_precision_at_20(model, loader, device="cuda"):
    model.eval()
    all_ap = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            topk_indices = torch.argsort(probs, dim=1, descending=True)[:, :20]

            for i in range(labels.size(0)):
                true_labels = labels[i].nonzero(as_tuple=True)[0]
                if len(true_labels) == 0:
                    continue

                precision_sum = 0.0
                hits = 0
                for k, idx in enumerate(topk_indices[i]):
                    if idx in true_labels:
                        hits += 1
                        precision_at_k = hits / (k + 1)
                        precision_sum += precision_at_k

                ap = precision_sum / min(len(true_labels), 20)
                all_ap.append(ap)

    return float(np.mean(all_ap)) if len(all_ap) > 0 else 0.0

#Training and Eval

# Can comment the line to evaluate map@20 on train if we want to save time 
def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    lr=1e-3,
    num_epochs=3,
    model_save_path=MODEL_SAVE_PATH,
    csv_log_path=CSV_LOG_PATH,
    device="cuda"
):
    """
    Train DenseNet for multi-label classification and evaluate with MAP@20.
    Saves the model and logs to CSV, also plots progress.
    """
    # Send model to GPU if available
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For plotting/logging
    epoch_list = []
    train_loss_list = []
    test_map_list = []
    
    # CSV columns
    columns = ["epoch", "train_loss", "test_map20"]
    if not os.path.exists(csv_log_path):
        pd.DataFrame(columns=columns).to_csv(csv_log_path, index=False)
    
    for epoch in range(1, num_epochs + 1):
        print(f"Starting Epoch #{epoch}")
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):

            if (batch_idx+1) % 30 == 0:
                print(f"Batch Number {batch_idx+1}")

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        if epoch % 5 == 0:

            # Evaluate MAP@20 on test set
            test_map20 = mean_average_precision_at_20(model, test_loader, device=device)
            
            print(f"Epoch {epoch}/{num_epochs}, "
                f"Train Loss: {avg_loss:.4f}, "
                f"Test MAP@20: {test_map20:.4f}")
            
            epoch_list.append(epoch)
            train_loss_list.append(avg_loss)
            test_map_list.append(test_map20)
            
            # Append to CSV
            df_log = pd.DataFrame([[epoch, avg_loss, test_map20]], columns=columns)
            df_log.to_csv(csv_log_path, mode='a', header=False, index=False)
    
            # Save the model weights
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")


def run_experiment(num_epochs=3, lr=1e-3, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # can tweak size and ratio
        transforms.ToTensor(),          # scales pixel values to [0,1]
    ])
    dataset = MultiLabelImageDataset(csv_file=LABELS_PATH, img_dir=IMAGES_PATH, transform=transform)
    print("Dataset has been loaded!")
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train set: {len(train_dataset)}, Test set: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = create_densenet(num_classes=290, pretrained=True)
    print("DenseNet model created.")
    
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=lr,
        num_epochs=num_epochs,
        model_save_path=MODEL_SAVE_PATH,
        csv_log_path=CSV_LOG_PATH,
        device=device
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on cuda")

run_experiment(num_epochs=50, lr=1e-3, device=device)