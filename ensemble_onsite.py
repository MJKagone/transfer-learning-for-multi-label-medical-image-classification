import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from src import dataset, models 


swin_path = "checkpoints/kagone_task4-swin.pt"
resnetSE_path = "checkpoints/kagone_task3-1.pt" 
output_file = "submission_ensemble.csv"
test_csv = "data/onsite_test_submission.csv"
test_image_dir = "./images/onsite_test"
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
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # Apply Sigmoid to get probabilities (0.0 to 1.0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            
    return np.array(all_probs)

# Get predictions from models
probs_swin = get_preds(swin_path, backbone="swin", attention=None)
probs_resnetSE = get_preds(resnetSE_path, backbone="resnet", attention="se")

# Soft voting for ensemble
print("Averaging predictions...")
avg_probs = (probs_swin + probs_resnetSE) / 2.0
preds = (avg_probs > 0.5).astype(int)

# Generate submission file
df = pd.read_csv(test_csv)
submission = pd.DataFrame({
    "id": df["id"],
    "D": preds[:, 0],
    "G": preds[:, 1],
    "A": preds[:, 2]
})

print(f"Saving ensemble predictions to {output_file}...")
submission.to_csv(output_file, index=False)