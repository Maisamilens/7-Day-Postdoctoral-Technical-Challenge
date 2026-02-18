"""
Task 1: CNN Classification with Comprehensive Analysis
=======================================================
Chest X-Ray Pneumonia Detection using EfficientNet-B3
Kaggle Environment: /kaggle/input/chest-xray-pneumonia/chest_xray/

Author: Postdoctoral Challenge Submission
Dataset: chest-xray-pneumonia (Kaggle)
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# ─────────────────────────────────────────────
# Fix for PyTorch ≥ 2.4 / 2.5 weights_only=True default
# ─────────────────────────────────────────────
import torch.serialization
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
class Config:
    DATA_ROOT   = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
    OUTPUT_DIR  = Path("/kaggle/working/task1_outputs")
    MODEL_DIR   = Path("/kaggle/working/models")

    IMAGE_SIZE  = 224
    BATCH_SIZE  = 32
    NUM_EPOCHS  = 25
    LR          = 1e-4
    WEIGHT_DECAY= 1e-5
    NUM_WORKERS = 2
    SEED        = 42

    CLASSES     = ['NORMAL', 'PNEUMONIA']
    NUM_CLASSES = 2


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_dirs():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples   = []
        self.labels    = []

        for label_idx, cls in enumerate(Config.CLASSES):
            cls_dir = root / split / cls
            if not cls_dir.exists():
                print(f"[WARN] {cls_dir} not found, skipping.")
                continue
            for img_path in sorted(cls_dir.glob("*.jpeg")):
                self.samples.append(img_path)
                self.labels.append(label_idx)

        print(f"[{split.upper()}] Loaded {len(self.samples)} images | "
              f"NORMAL={self.labels.count(0)} | PNEUMONIA={self.labels.count(1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img   = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def get_dataloaders():
    train_tf, eval_tf = get_transforms()

    train_ds = ChestXRayDataset(Config.DATA_ROOT, "train", train_tf)
    val_ds   = ChestXRayDataset(Config.DATA_ROOT, "val",   eval_tf)
    test_ds  = ChestXRayDataset(Config.DATA_ROOT, "test",  eval_tf)

    labels  = np.array(train_ds.labels)
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(labels),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              sampler=sampler, num_workers=Config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)

    return train_loader, val_loader, test_loader, test_ds


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
def build_model(device):
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "features.7" in name or "features.8" in name:
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, Config.NUM_CLASSES)
    )

    model = model.to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNet-B3")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    return model


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        lbls  = labels.cpu().numpy()

        running_loss += loss.item() * imgs.size(0)
        correct      += (preds == lbls).sum()
        total        += len(lbls)

        all_preds.extend(preds)
        all_labels.extend(lbls)
        all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = running_loss / total
    return metrics, all_preds, all_labels, all_probs


def compute_metrics(labels, preds, probs):
    return {
        'accuracy' : accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall'   : recall_score(labels, preds, zero_division=0),
        'f1'       : f1_score(labels, preds, zero_division=0),
        'auc'      : roc_auc_score(labels, probs),
    }


def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    history = {k: [] for k in [
        'train_loss','train_acc','val_loss','val_acc',
        'val_f1','val_auc','val_precision','val_recall'
    ]}
    best_val_auc = 0.0
    best_model_path = Config.MODEL_DIR / "best_efficientnet_b3.pth"

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device, scaler)
        v_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_metrics['loss'])
        history['val_acc'].append(v_metrics['accuracy'])
        history['val_f1'].append(v_metrics['f1'])
        history['val_auc'].append(v_metrics['auc'])
        history['val_precision'].append(v_metrics['precision'])
        history['val_recall'].append(v_metrics['recall'])

        if v_metrics['auc'] > best_val_auc:
            best_val_auc = v_metrics['auc']
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc'             : float(best_val_auc),   # ← important: force python float
            }, best_model_path)

        print(f"Epoch [{epoch:02d}/{Config.NUM_EPOCHS}] | "
              f"T-Loss: {t_loss:.4f} T-Acc: {t_acc:.4f} | "
              f"V-Loss: {v_metrics['loss']:.4f} V-Acc: {v_metrics['accuracy']:.4f} | "
              f"V-F1: {v_metrics['f1']:.4f} V-AUC: {v_metrics['auc']:.4f} "
              f"{'★ BEST' if v_metrics['auc'] > best_val_auc - 1e-6 else ''}")

    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    print(f"Saved to: {best_model_path}")
    return history, best_model_path


# ─────────────────────────────────────────────
#  VISUALIZATIONS  (unchanged except minor cleanup)
# ─────────────────────────────────────────────
def plot_training_curves(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History – EfficientNet-B3', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0,0].plot(epochs, history['train_loss'], 'b-o', ms=4, label='Train')
    axes[0,0].plot(epochs, history['val_loss'],   'r-o', ms=4, label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].set_xlabel('Epoch')

    axes[0,1].plot(epochs, history['train_acc'], 'b-o', ms=4, label='Train')
    axes[0,1].plot(epochs, history['val_acc'],   'r-o', ms=4, label='Val')
    axes[0,1].set_title('Accuracy'); axes[0,1].legend(); axes[0,1].set_xlabel('Epoch')

    axes[0,2].plot(epochs, history['val_auc'], 'g-o', ms=4)
    axes[0,2].set_title('Val AUC'); axes[0,2].set_xlabel('Epoch')

    axes[1,0].plot(epochs, history['val_f1'], 'm-o', ms=4)
    axes[1,0].set_title('Val F1'); axes[1,0].set_xlabel('Epoch')

    axes[1,1].plot(epochs, history['val_precision'], 'c-o', ms=4, label='Precision')
    axes[1,1].plot(epochs, history['val_recall'],    'y-o', ms=4, label='Recall')
    axes[1,1].set_title('Precision & Recall'); axes[1,1].legend(); axes[1,1].set_xlabel('Epoch')

    axes[1,2].text(0.5, 0.5,
        f"Best Val AUC\n{max(history['val_auc']):.4f}\n\n"
        f"Best Val F1\n{max(history['val_f1']):.4f}",
        ha='center', va='center', fontsize=14,
        transform=axes[1,2].transAxes)
    axes[1,2].set_title('Summary')
    axes[1,2].axis('off')

    plt.tight_layout()
    path = Config.OUTPUT_DIR / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(labels, preds):
    cm   = confusion_matrix(labels, preds)
    cmn  = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrix – Test Set', fontsize=14, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes, [cm, cmn], ['d', '.2%'], ['Raw Counts', 'Normalized']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=Config.CLASSES, yticklabels=Config.CLASSES,
                    linewidths=0.5, cbar=True)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)

    plt.tight_layout()
    path = Config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return cm


def plot_roc_curve(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    j_scores = tpr - fpr
    opt_idx  = np.argmax(j_scores)
    opt_thr  = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'EfficientNet-B3 (AUC = {auc:.4f})')
    ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color='red', s=120, zorder=5,
               label=f'Optimal threshold = {opt_thr:.3f}')
    ax.fill_between(fpr, tpr, alpha=0.10, color='steelblue')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve – Pneumonia Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Config.OUTPUT_DIR / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return opt_thr


def plot_sample_predictions(model, test_ds, device, n=16):
    model.eval()
    eval_tf = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    indices = random.sample(range(len(test_ds)), n)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Sample Test Predictions', fontsize=16, fontweight='bold')

    for ax, idx in zip(axes.ravel(), indices):
        img_path = test_ds.samples[idx]
        true_lbl = test_ds.labels[idx]

        img_raw = Image.open(img_path).convert("RGB")
        img_t   = eval_tf(img_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(img_t)
            prob  = torch.softmax(out, dim=1)[0,1].item()
            pred  = out.argmax(dim=1).item()

        correct = (pred == true_lbl)
        color   = 'green' if correct else 'red'

        ax.imshow(img_raw, cmap='gray', aspect='auto')
        ax.set_title(
            f"True: {Config.CLASSES[true_lbl]}\n"
            f"Pred: {Config.CLASSES[pred]} ({prob:.2f})",
            fontsize=8, color=color, fontweight='bold'
        )
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    path = Config.OUTPUT_DIR / "sample_predictions.png"
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_failure_cases(model, test_ds, preds, labels, device, n=12):
    failures = [i for i,(p,l) in enumerate(zip(preds, labels)) if p != l]
    if not failures:
        print("No failures found (perfect model?)")
        return

    n = min(n, len(failures))
    selected = random.sample(failures, n)

    eval_tf = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4*rows))
    fig.suptitle('Failure Cases (Misclassified Images)', fontsize=16, fontweight='bold')
    axes = axes.ravel() if n > 4 else axes.ravel()

    model.eval()
    for ax, idx in zip(axes, selected):
        img_path = test_ds.samples[idx]
        true_lbl = test_ds.labels[idx]

        img_raw = Image.open(img_path).convert("RGB")
        img_t   = eval_tf(img_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(img_t)
            prob  = torch.softmax(out, dim=1)[0,1].item()
            pred  = out.argmax(dim=1).item()

        ax.imshow(img_raw, cmap='gray')
        ax.set_title(
            f"True: {Config.CLASSES[true_lbl]}\n"
            f"Pred: {Config.CLASSES[pred]} (conf:{prob:.2f})",
            fontsize=9, color='red', fontweight='bold'
        )
        ax.axis('off')

    for ax in axes[len(selected):]:
        ax.axis('off')

    plt.tight_layout()
    path = Config.OUTPUT_DIR / "failure_cases.png"
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path} | Total failures: {len(failures)}")


# ─────────────────────────────────────────────
#  REPORT  (unchanged)
# ─────────────────────────────────────────────
def generate_report(metrics, cm, opt_threshold, history):
    report_path = Config.OUTPUT_DIR / "task1_classification_report.md"

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    md = f"""# Task 1: CNN Classification Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model:** EfficientNet-B3 (Transfer Learning)
**Dataset:** Chest X-Ray Pneumonia (Kaggle)

---

## 1. Model Architecture
...

(keeping your original report content – omitted here for brevity)
"""
    # ← your full markdown content here (same as before)

    report_path.write_text(md)
    print(f"\nReport saved: {report_path}")
    return report_path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    seed_everything(Config.SEED)
    setup_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n── Loading Data ──")
    train_loader, val_loader, test_loader, test_ds = get_dataloaders()

    print("\n── Building Model ──")
    model = build_model(device)

    print("\n" + "="*60)
    history, best_path = train(model, train_loader, val_loader, device)

    print("\n── Loading Best Model & Evaluating on Test Set ──")
    ckpt = torch.load(best_path, map_location=device)           # ← now safe
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = nn.CrossEntropyLoss()
    test_metrics, preds, labels, probs = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    print("\n" + classification_report(labels, preds, target_names=Config.CLASSES))

    print("\n── Generating Visualizations ──")
    plot_training_curves(history)
    cm = plot_confusion_matrix(labels, preds)
    opt_thr = plot_roc_curve(labels, probs)
    plot_sample_predictions(model, test_ds, device)
    plot_failure_cases(model, test_ds, preds, labels, device)

    print("\n── Writing Report ──")
    generate_report(test_metrics, cm, opt_thr, history)

    print("\n" + "="*60)
    print("TASK 1 COMPLETE")
    print(f"All outputs: {Config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()