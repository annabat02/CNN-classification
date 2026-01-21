import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =======================
# 1. Device setup
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# 2. Config & Transform (ResNet Specific)
# =======================
DATASET_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# ResNet butuh 224x224 dan normalisasi ImageNet
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# =======================
# 3. Data Splitting (75-15-10)
# =======================
print("=== Memulai Data Splitting... ===")
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

all_data = []
for cls_name in classes:
    cls_folder = os.path.join(DATASET_DIR, cls_name)
    for img_name in os.listdir(cls_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            all_data.append({
                "path": os.path.join(cls_folder, img_name),
                "image_name": img_name,
                "label_name": cls_name,
                "label_idx": class_to_idx[cls_name]
            })

df_all = pd.DataFrame(all_data)

# Split 1: 75% Train
train_df, temp_df = train_test_split(df_all, train_size=0.75, stratify=df_all['label_idx'], random_state=42)
# Split 2: Sisa 25% dibagi -> 15% Test (60% dari sisa) & 10% Val (40% dari sisa)
val_df, test_df = train_test_split(temp_df, test_size=0.6, stratify=temp_df['label_idx'], random_state=42)

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Save Distribution CSV
pd.concat([train_df, val_df, test_df]).to_csv("data_distribution_resnet.csv", index=False)
print(f"✅ Data distribution saved. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# =======================
# 4. Custom Dataset
# =======================
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['path']).convert("RGB")
        label = row['label_idx']
        if self.transform:
            image = self.transform(image)
        return image, label, row['image_name']

train_loader = DataLoader(CustomImageDataset(train_df, transform_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CustomImageDataset(val_df, transform_test), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CustomImageDataset(test_df, transform_test), batch_size=1, shuffle=False)

# =======================
# 5. Model Definition (ResNet18)
# =======================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# Freeze layer selain FC
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# =======================
# 6. Training Loop
# =======================
total_start_time = time.time()
epoch_times = []

print("\n=== START TRAINING RESNET18 ===")
for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    # Train
    model.train()
    train_loss = 0.0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # Val
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    
    epoch_duration = time.time() - epoch_start_time
    epoch_times.append(epoch_duration)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_duration:.2f}s")

# Time Summary
print("\n=== TRAINING TIME SUMMARY ===")
print(f"Total: {time.time() - total_start_time:.2f}s | Avg/Epoch: {sum(epoch_times)/len(epoch_times):.2f}s")

# =======================
# 7. Testing & Evaluation
# =======================
print("\n=== START TESTING ===")
model.eval()
y_true, y_pred, probs_all, names_list = [], [], [], []

with torch.no_grad():
    for imgs, labels, names in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        
        y_true.append(labels.item())
        y_pred.append(preds.item())
        probs_all.append(probs.cpu().numpy()[0])
        names_list.append(names[0])

# CSV Prediksi
df_pred = pd.DataFrame({
    "image_name": names_list,
    "y_true": y_true,
    "y_pred": y_pred,
    "true_label": [classes[i] for i in y_true],
    "pred_label": [classes[i] for i in y_pred]
})
df_pred.to_csv("hasil_prediksi_resnet.csv", index=False)

# Metrik
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
mse = mean_squared_error(np.eye(len(classes))[y_true], probs_all)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Pixel Accuracy", "MSE"],
    "Value": [acc, prec, rec, f1, acc, mse]
})
print(metrics)
metrics.to_csv("evaluation_metrics_resnet.csv", index=False)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - ResNet18")
plt.savefig("confusion_matrix_resnet.png")
plt.show()

# =======================
# 8. Save Model
# =======================
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': class_to_idx,
    'idx_to_class': {v: k for k, v in class_to_idx.items()}
}, "resnet18_cuaca.pth")
print("✅ Model saved successfully!")