import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from src import dataset, models 

swin_path = "checkpoints/kagone_task4-swin.pt"
resnetSE_path = "checkpoints/kagone_task3-1.pt" 
output_file = "submission_ensemble.csv"
test_csv = "data/offsite_test.csv"
test_image_dir = "./images/offsite_test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_preds(ckpt_path, backbone, attention=None):
    print(f"--- Processing {backbone} (Attention: {attention}) ---")
    
    model = models.build_model(backbone, num_classes=3, pretrained=False, attention=attention)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    print(f"Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, trying strict=False. Error: {e}")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = dataset.RetinaMultiLabelDataset(test_csv, test_image_dir, val_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            # Get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    return np.array(all_probs), np.array(all_labels)


# Get predictions from models
probs_swin, y_true_swin = get_preds(swin_path, backbone="swin", attention=None)
probs_resnetSE, y_true_resnet = get_preds(resnetSE_path, backbone="resnet", attention="se")
y_true = y_true_swin

# Soft voting for ensemble
print("Averaging predictions...")
avg_probs = (probs_swin + probs_resnetSE) / 2.0
preds = (avg_probs > 0.5).astype(int)

# Report metrics
disease_names = ["DR", "Glaucoma", "AMD"]
f1s = []

for i, disease in enumerate(disease_names):
    y_t = y_true[:, i]
    y_p = preds[:, i]

    acc = accuracy_score(y_t, y_p)
    precision = precision_score(y_t, y_p, average="macro", zero_division=0)
    recall = recall_score(y_t, y_p, average="macro", zero_division=0)
    f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
    f1s.append(f1)
    kappa = cohen_kappa_score(y_t, y_p)

    print(f"{disease} Results [Ensemble]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")
    print("-" * 20)

print(f"Average F1-score: {np.mean(f1s):.4f}")