from src import dataset
from src import models
from src import utils

import argparse
import os
import numpy as np
import warnings

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

warnings.filterwarnings("ignore")

#################################
# Model training and validation #
#################################
def train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, fine_tuning,
                       loss, attention, epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints", pretrained_backbone=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, scale=(0.9,1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Clean transform for val/test
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = dataset.RetinaMultiLabelDataset(train_csv, train_image_dir, train_transform)
    
    val_ds   = dataset.RetinaMultiLabelDataset(val_csv, val_image_dir, val_transform)
    test_ds  = dataset.RetinaMultiLabelDataset(test_csv, test_image_dir, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.build_model(backbone, num_classes=3, pretrained=False, attention=attention).to(device)
    if backbone == "efficientnet" and fine_tuning == "full":
        model.classifier[0].p = 0.5 # increase dropout for fine-tuning

    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location=device)
        model.load_state_dict(state_dict, strict=False) 
        print(f"Loaded backbone from {pretrained_backbone}")

    if attention is not None or loss == "focal":
        lr = 5e-4  # override lr when using attention or focal loss

    if fine_tuning != "none":

        if fine_tuning == "full":
            for p in model.parameters():
                p.requires_grad = True
                
            # Different learning rates for backbone and head
            if backbone == "resnet":
                backbone_params = [p for name, p in model.named_parameters() if "fc" not in name and "avgpool" not in name]
                head_params = list(model.fc.parameters()) + list(model.avgpool.parameters())
            elif backbone == "efficientnet" or backbone == "mobilenet":
                backbone_params = [p for name, p in model.named_parameters() if "classifier" not in name and "avgpool" not in name]
                head_params = list(model.classifier.parameters()) + list(model.avgpool.parameters())
            elif backbone == "swin":
                lr = 2e-4  # override lr for swin
                backbone_params = [p for name, p in model.named_parameters() if "head" not in name]
                head_params = list(model.head.parameters())
            optimizer = optim.AdamW([
                {"params": backbone_params, 'lr': lr * 0.05},
                {"params": head_params, 'lr': lr}
            ])
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        elif fine_tuning == "classifier":
            for p in model.parameters():
                p.requires_grad = False
            if backbone == "resnet":
                for p in model.fc.parameters():
                    p.requires_grad = True
            elif backbone == "efficientnet":
                for p in model.classifier.parameters():
                    p.requires_grad = True
            
            lr = 1e-3 # override lr for classifier fine-tuning
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


        else:
            raise ValueError("Unsupported fine_tuning option")
        
        if loss == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif loss == "focal":
            from torchvision.ops import sigmoid_focal_loss
            def focal_loss_wrapper(inputs, targets):
                return sigmoid_focal_loss(inputs, targets, alpha=0.75, gamma=1.0, reduction="mean")
            criterion = focal_loss_wrapper
        elif loss == "balanced":
            class_weights = np.array([1.0, 3.17, 3.64])
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
        else:
            raise ValueError("Unsupported loss option")

        best_val_f1 = 0.0
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"best_{backbone}_{fine_tuning}.pt")
        train_loss_history = []
        val_loss_history = []
        val_f1_history = []

        for epoch in range(epochs):
            model.train()

            # If freezing backbone, force it to eval mode to freeze BN stats
            if fine_tuning == "classifier":
                model.eval()
                # Now unfreeze head
                if backbone == "resnet":
                    model.fc.train()
                elif backbone == "efficientnet":
                    model.classifier.train()

            train_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            if fine_tuning == "full" or fine_tuning == "classifier":
                scheduler.step()

            train_loss /= len(train_loader.dataset)
            train_loss_history.append(train_loss)

            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)

                    # Collect preds and targets for f1
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    val_preds.extend(preds)
                    val_targets.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)

            f1s_epoch = []
            for i in range(3):
                f1 = f1_score(val_targets[:, i], val_preds[:, i], average="macro", zero_division=0)
                f1s_epoch.append(f1)
            current_val_f1 = np.mean(f1s_epoch)
            val_f1_history.append(current_val_f1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val F1: {current_val_f1:.4f}")

            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                torch.save(model.state_dict(), ckpt_path)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, epochs+1), train_loss_history, label="Train Loss")
        plt.plot(range(1, epochs+1), val_loss_history, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {backbone} ({fine_tuning})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, f"loss_curve_{backbone}_{fine_tuning}.png"))
        plt.close()

    #########################
    # TESTING AND REPORTING #
    #########################
    if fine_tuning != "none":
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best model for {backbone} from {ckpt_path}")
    else:
        print(f"Evaluating {backbone} without fine-tuning")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    disease_names = ["DR", "Glaucoma", "AMD"]
    f1s = []

    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro",zero_division=0)
        recall = recall_score(y_t, y_p, average="macro",zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro",zero_division=0)
        f1s.append(f1)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")

    if fine_tuning == "none" and backbone == "resnet":
        print("Average F1-score: {:.4f} (reference: 0.567)".format(np.mean(f1s)))
    elif fine_tuning == "classifier" and backbone == "resnet":
        print("Average F1-score: {:.4f} (reference: 0.614)".format(np.mean(f1s)))
    elif fine_tuning == "full" and backbone == "resnet":
        print("Average F1-score: {:.4f} (reference: 0.788)".format(np.mean(f1s)))
    elif fine_tuning == "none" and backbone == "efficientnet":
        print("Average F1-score: {:.4f} (reference: 0.604)".format(np.mean(f1s)))
    elif fine_tuning == "classifier" and backbone == "efficientnet":
        print("Average F1-score: {:.4f} (reference: 0.735)".format(np.mean(f1s)))
    elif fine_tuning == "full" and backbone == "efficientnet":
        print("Average F1-score: {:.4f} (reference: 0.804)".format(np.mean(f1s)))
    elif fine_tuning == "full" and backbone == "mobilenet":
        print("Average F1-score: {:.4f} (reference: N/A)".format(np.mean(f1s)))
    elif fine_tuning == "full" and backbone == "swin":
        print("Average F1-score: {:.4f} (reference: N/A)".format(np.mean(f1s)))
    else:
        print("Something went wrong in reporting results.")
    
if __name__ == "__main__":
    utils.seed_everything(67)

    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuning", type=str, default="full", 
                        choices=["none", "classifier", "full"], help="Fine-tuning mode")
    parser.add_argument("--backbone", type=str),
    parser.add_argument("--loss", type=str, default="bce", choices=["bce","focal","balanced"], help="Loss function for training")
    parser.add_argument("--attention", type=str, default=None, 
                    choices=["se", "mha"], help="Add attention mechanism (Task 3)")
    args = parser.parse_args()

    train_csv = "data/train.csv"
    val_csv   = "data/val.csv"
    test_csv  = "data/offsite_test.csv"
    train_image_dir ="./images/train"
    val_image_dir = "./images/val"
    test_image_dir = "./images/offsite_test"
    pretrained_path = './pretrained_backbone/ckpt_resnet18_ep50.pt'

    if args.backbone == "resnet":
        pretrained_path = "./pretrained_backbone/ckpt_resnet18_ep50.pt"
    elif args.backbone == "efficientnet":
        pretrained_path = "./pretrained_backbone/ckpt_efficientnet_ep50.pt"
    else:
        pretrained_path = None

    train_one_backbone(
        backbone=args.backbone,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        epochs=50,
        batch_size=32,
        lr=3e-4,
        img_size=256,
        pretrained_backbone=pretrained_path,
        fine_tuning=args.fine_tuning,
        loss=args.loss,
        attention=args.attention   
    )