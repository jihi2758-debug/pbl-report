import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PATHS
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "resnet50_attention_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
BATCH_SIZE = 16

print("Device:", DEVICE)

# =========================
# TEST TRANSFORM
# =========================

test_tf = A.Compose([
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
# LOAD TEST DATA
# =========================

test_ds = AlbDataset(os.path.join(DATA_DIR,"test"), test_tf)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Test samples:", len(test_ds))

# =========================
# ATTENTION MODULE (CBAM)
# =========================

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        max,_ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg,max], dim=1)

        x = self.conv(x)

        return self.sigmoid(x)

class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):

        x = x * self.ca(x)
        x = x * self.sa(x)

        return x

# =========================
# MODEL
# =========================

class ResNet50_Attention(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        base_model = models.resnet50(weights=None)

        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        self.attention = CBAM(2048)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self,x):

        x = self.features(x)

        x = self.attention(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x

# =========================
# LOAD MODEL
# =========================

model = ResNet50_Attention(NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)

model.eval()

print("Model loaded successfully")

# =========================
# TEST EVALUATION
# =========================

all_preds = []
all_labels = []
all_probs = []

correct = 0
total = 0

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_acc = 100 * correct / total

print("\nTest Accuracy:", test_acc)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Test Confusion Matrix")

plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report\n")

print(classification_report(all_labels, all_preds))

# =========================
# ROC CURVE
# =========================

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

plt.figure()

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()