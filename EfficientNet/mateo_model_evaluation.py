import torch
import torch.nn as nn
from mateo_model import EfficientNetModel
from torch.utils.data import DataLoader, random_split

def setup_model_and_data(dataset_path='dataset.pt', model_weights_path='efficientnet_model_weights.pth', num_classes=290, batch_size=32):
    # Load dataset and split (same seed as training)
    dataset = torch.load(dataset_path)
    print("Dataset loaded")
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Dataset split into train ({train_size}) and test ({test_size})")

    # Set up model
    model = EfficientNetModel(num_classes)
    print("Model created")

    # Load model weights
    state_dict_half = torch.load(model_weights_path, map_location="cpu")
    state_dict_fp32 = {k: v.float() for k, v in state_dict_half.items()}  # Convert to FP32
    model.load_state_dict(state_dict_fp32)
    model.eval()  # Set model to evaluation mode
    print("Model weights loaded")

    # Set batch size for evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return model, test_loader

# Accuracy evaluation
def evaluate_accuracy(model, loader, threshold=0.5):
    model.eval()
    correct = 0  
    total_labels = 0
    batch_count = 0

    with torch.no_grad():
        for images, labels in loader:
            batch_count += 1
            print(f"Processing Batch #{batch_count}")

            outputs = model(images)
            preds = torch.sigmoid(outputs) >= threshold  # Convert logits to probabilities

            correct += (preds == labels.byte()).sum().item()  # Sum correct predictions
            total_labels += labels.numel()

    accuracy = correct / total_labels
    return accuracy

# Mean Average Precision @ 20
def mean_average_precision_at_20(model, loader):
    model.eval()
    all_ap = []
    batch_count = 0

    with torch.no_grad():
        for images, labels in loader:
            batch_count += 1
            print(f"Processing Batch #{batch_count}")

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            topk_indices = torch.argsort(probs, dim=1, descending=True)[:, :20]

            for i in range(labels.shape[0]):  # Iterate over batch
                relevant_labels = labels[i].nonzero(as_tuple=True)[0]

                if len(relevant_labels) == 0:
                    continue  # Skip if no relevant labels

                num_relevant = len(relevant_labels)
                precision_sum = 0.0
                num_hits = 0  

                for k, idx in enumerate(topk_indices[i]):
                    if idx in relevant_labels:
                        num_hits += 1
                        precision_at_k = num_hits / (k + 1)
                        precision_sum += precision_at_k

                ap = precision_sum / min(num_relevant, 20)
                all_ap.append(ap)

    return sum(all_ap) / len(all_ap) if all_ap else 0.0  

# Run evaluations
if __name__ == "__main__":
    model, test_loader = setup_model_and_data()
    print("Evaluating Accuracy...")
    test_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Test per-label accuracy: {test_accuracy:.4f}")

    print("Evaluating mAP@20...")
    test_map20 = mean_average_precision_at_20(model, test_loader)
    print(f"Test mAP@20: {test_map20:.4f}")
