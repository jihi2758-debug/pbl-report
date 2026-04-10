import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PATHS & CONFIG
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3

print("Device:", DEVICE)
print("Dataset path:", DATA_DIR)

# =========================
# TRANSFORMS
# =========================

train_tf = A.Compose([

    # CLAHE improves contrast in X-ray
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),

    A.Resize(224,224),

    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.04,
        scale_limit=0.10,
        rotate_limit=10,
        p=0.7
    ),

    A.RandomBrightnessContrast(p=0.4),

    A.GaussianBlur(p=0.2),

    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),

    ToTensorV2(),
])


val_tf = A.Compose([

    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),

    A.Resize(224,224),

    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),

    ToTensorV2(),
])

# =========================
# DATASET CLASS
# =========================

class AlbDataset(Dataset):

    def __init__(self, folder, transform):

        self.paths = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(folder))

        for label in classes:

            class_dir = os.path.join(folder, label)

            for img in os.listdir(class_dir):

                if img.lower().endswith((".png",".jpg",".jpeg",".bmp")):

                    self.paths.append(os.path.join(class_dir,img))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = np.array(Image.open(self.paths[idx]).convert("RGB"))

        img = self.transform(image=img)["image"]

        return img, self.labels[idx]

# =========================
# LOAD DATA
# =========================

train_ds = AlbDataset(os.path.join(DATA_DIR,"train"), train_tf)
val_ds = AlbDataset(os.path.join(DATA_DIR,"val"), val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# =========================
# CLASS WEIGHTS
# =========================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_ds.labels),
    y=train_ds.labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print("Class weights:", class_weights)

# =========================
# MODEL
# =========================

model = models.resnet50(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)

model.to(DEVICE)

# =========================
# LOSS / OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,
    patience=5
)

scaler = GradScaler("cuda")

# =========================
# METRIC STORAGE
# =========================

train_losses = []
val_losses = []

train_accs = []
val_accs = []

# =========================
# TRAINING LOOP
# =========================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast("cuda"):

            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            probs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    scheduler.step(val_acc)

    train_losses.append(total_loss)
    val_losses.append(val_loss)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Val Acc {val_acc:.2f}%"
    )

# =========================
# SAVE MODEL
# =========================

model_path = os.path.join(SCRIPT_DIR,"resnet50_clahe_model.pth")

torch.save(model.state_dict(), model_path)

print("Model saved to:", model_path)

# =========================
# PLOT ACCURACY
# =========================

epochs_range = range(1, EPOCHS+1)

plt.figure()
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid()

plt.savefig(os.path.join(SCRIPT_DIR,"accuracy_plot.png"))

# =========================
# PLOT LOSS
# =========================

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid()

plt.savefig(os.path.join(SCRIPT_DIR,"loss_plot.png"))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(os.path.join(SCRIPT_DIR,"confusion_matrix.png"))

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report\n")

print(classification_report(all_labels, all_preds))

# =========================
# ROC CURVE (MULTI CLASS)
# =========================

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

plt.figure()

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (OvR)")
plt.legend()

plt.savefig(os.path.join(SCRIPT_DIR,"roc_curve.png"))