import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import ast
import numpy as np

########################################
# Helper: Parse category list from CSV
########################################
def parse_categories(cat_str):
    try:
        cats = ast.literal_eval(cat_str)
        if not isinstance(cats, list):
            raise ValueError(f"Expected list but got {type(cats)}: {cats}")
        return [int(x) for x in cats]
    except Exception as e:
        print(f"Error parsing categories from {cat_str}: {e}")
        return []

########################################
# Helper: Create multi-hot vector (not used for inference display)
########################################
def multi_hot_vector(categories, num_classes=290):
    vector = np.zeros(num_classes, dtype=np.float32)
    for cat in categories:
        index = int(cat) - 1  # adjust for zero-indexing
        if 0 <= index < num_classes:
            vector[index] = 1.0
    return vector

########################################
# Main inference function using DenseNet
########################################
def run_inference(
    csv_file="data/train_classification_labels.csv",
    img_dir="data/train", 
    image_id="2e396e05-3583-41ae-b6ec-37cb7120a8b9.png", 
    model_weights="densenet_model.pth",
    category_key_csv="data/category_key.csv",
    num_classes=290,
    topk=5
):

    # Step A: Load category key (mapping id -> (name, supercat))
    category_key_df = pd.read_csv(category_key_csv)
    category_dict = {}
    for _, row in category_key_df.iterrows():
        cat_id = row["id"]
        cat_name = row["name"]
        supercat = row["supercat"]
        category_dict[cat_id] = (cat_name, supercat)

    # Step B: Load the CSV row for this image
    df = pd.read_csv(csv_file)
    record = df[df["id"] == image_id.replace(".png", "")]
    if record.empty:
        print(f"No CSV row found for image ID: {image_id}")
        return
    cat_str = record.iloc[0]["categories"]
    categories = parse_categories(cat_str)
    print(f"Parsed categories for image {image_id}: {categories}")

    # Step C: Load and transform the image (matching your training transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_path = os.path.join(img_dir, image_id)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Step D: Define and load the DenseNet model
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu")))
    model.eval()

    # Step E: Run inference and get top 5 predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)

    topk_values, topk_indices = torch.topk(probs.squeeze(0), topk)
    topk_indices = topk_indices.tolist()
    topk_values = topk_values.tolist()
    predicted_categories = [idx + 1 for idx in topk_indices]

    print("=== Model Top-5 Predictions ===")
    for rank, (cat_id, p_val) in enumerate(zip(predicted_categories, topk_values), 1):
        cat_name, supercat = category_dict.get(cat_id, ("Unknown", "Unknown"))
        print(f"{rank}. ID: {cat_id}, Name: {cat_name}, Supercat: {supercat}, Probability: {p_val:.4f}")

    print("\n=== Ground-Truth Categories ===")
    for cat_id in categories:
        cat_name, supercat = category_dict.get(cat_id, ("Unknown", "Unknown"))
        print(f"  ID: {cat_id}, Name: {cat_name}, Supercat: {supercat}")



if __name__ == "__main__":
    run_inference(
        csv_file="data/train_classification_labels.csv",
        img_dir="data/train",
        image_id="2e396e05-3583-41ae-b6ec-37cb7120a8b9.png",
        model_weights="densenet_model.pth",
        category_key_csv="data/category_key.csv",
        num_classes=290,
        topk=5
    )