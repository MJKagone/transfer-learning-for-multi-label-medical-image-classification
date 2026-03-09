import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from src import dataset
from src import models

test_img_dir = "images/onsite_test"
test_csv = "data/onsite_test_submission.csv"

def generate_predictions(model_name, test_csv, test_img_dir, fine_tuning, attention=None, batch_size=32, img_size=256, output_csv="submission.csv"):

    if fine_tuning == "none" and model_name == "resnet":
        model_path = "pretrained_backbone/ckpt_resnet18_ep50.pt"
    elif fine_tuning == "none" and model_name == "efficientnet":
        model_path = "pretrained_backbone/ckpt_efficientnet_ep50.pt"
    elif fine_tuning == "classifier" and model_name == "resnet":
        model_path = "checkpoints/best_resnet_classifier.pt"
    elif fine_tuning == "classifier" and model_name == "efficientnet":
        model_path = "checkpoints/best_efficientnet_classifier.pt"
    elif fine_tuning == "full" and model_name == "resnet":
        model_path = "checkpoints/best_resnet_full.pt"
    elif fine_tuning == "full" and model_name == "efficientnet":
        model_path = "checkpoints/best_efficientnet_full.pt"
    elif fine_tuning == "full" and model_name == "swin":
        model_path = "checkpoints/best_swin_full.pt"
    elif fine_tuning == "full" and model_name == "mobilenet":
        model_path = "checkpoints/best_mobilenet_full.pt"
    else:
        raise ValueError("Unsupported combination of model and fine-tuning strategy.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use clean transform for testing
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    test_ds = dataset.RetinaMultiLabelDataset(test_csv, test_img_dir, test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.build_model(backbone=model_name, num_classes=3, pretrained=False, attention=attention)
    
    print(f"Loading weights from {model_path} with attention={attention}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []

    print(f"Generating predictions...")
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # Get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)

    all_preds = np.vstack(all_preds)
    binary_preds = (all_preds > 0.5).astype(int)

    ids = pd.read_csv(test_csv).iloc[:, 0].values
    submission_df = pd.DataFrame(binary_preds, columns=["D", "G", "A"])
    submission_df.insert(0, "id", ids)

    submission_df.to_csv(output_csv, index=False)
    print(f"Submission file saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", help="Model backbone: resnet or efficientnet")
    parser.add_argument("--fine_tuning", type=str, default="full", help="Fine-tuning strategy: none, classifier or full")
    parser.add_argument("--attention", type=str, default=None, choices=["se", "mha"], help="Attention mechanism used in training")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Output submission CSV file")
    args = parser.parse_args()

    generate_predictions(args.model, test_csv, test_img_dir, fine_tuning=args.fine_tuning, attention=args.attention, output_csv=args.output_csv)