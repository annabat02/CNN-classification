import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
# 2. Config & Transform
# =======================
DATASET_DIR = "data" # Pastikan nama folder dataset kamu benar
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# =======================
# 3. Data Splitting Strategy (75-15-10) & CSV Generation
# =======================
print("=== Memulai Data Splitting... ===")

# Ambil semua kelas
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("Classes:", classes)

all_data = []

# Loop semua file
for cls_name in classes:
    cls_folder = os.path.join(DATASET_DIR, cls_name)
    for img_name in os.listdir(cls_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(cls_folder, img_name)
            all_data.append({
                "path": full_path,
                "image_name": img_name,
                "label_name": cls_name,
                "label_idx": class_to_idx[cls_name]
            })

df_all = pd.DataFrame(all_data)

# Split 1: Pisahkan Train (75%) dan Sisa (25%)
train_df, temp_df = train_test_split(df_all, train_size=0.75, stratify=df_all['label_idx'], random_state=42)

# Split 2: Pisahkan Sisa (25%) menjadi Test (15% total) dan Val (10% total)
# Karena 15% adalah 60% dari 25%, dan 10% adalah 40% dari 25%
val_df, test_df = train_test_split(temp_df, test_size=0.6, stratify=temp_df['label_idx'], random_state=42)

# Beri tag untuk CSV distribusi
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Gabungkan dan simpan distribusi
distribution_df = pd.concat([train_df, val_df, test_df])
distribution_df.to_csv("data_distribution.csv", index=False)
print(f"✅ Data distribution saved to 'data_distribution.csv'")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# =======================
# 4. Custom Dataset Class (Updated)
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
        
        return image, label, row['image_name'] # Return nama image juga untuk tracking

# Buat Dataset Objects
train_dataset = CustomImageDataset(train_df, transform=transform_train)
val_dataset = CustomImageDataset(val_df, transform=transform_test)
test_dataset = CustomImageDataset(test_df, transform=transform_test)

# Buat DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 untuk testing per item

# =======================
# 5. Model Definition (Simple CNN)
# =======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Menghitung ukuran input linear secara dinamis berdasarkan input size 128x128
        # 128 -> 64 -> 32 -> 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =======================
# 6. Training Loop
# =======================
total_start_time = time.time()
epoch_times = []

print("\n=== START TRAINING ===")
for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    # -------- TRAINING --------
    model.train()
    train_loss = 0.0
    for imgs, labels, _ in train_loader: # _ adalah img_names (tidak dipakai saat training)
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # -------- VALIDATION --------
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

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Time: {epoch_duration:.2f}s"
    )

total_end_time = time.time()
total_training_time = total_end_time - total_start_time
avg_epoch_time = sum(epoch_times) / len(epoch_times)

print("\n=== TRAINING TIME SUMMARY ===")
print(f"Total Training Time   : {total_training_time:.2f} seconds")
print(f"Average Time / Epoch  : {avg_epoch_time:.2f} seconds")

# =======================
# 7. Testing & Evaluation Metrics
# =======================
print("\n=== START TESTING ===")
model.eval()
y_true, y_pred, probs_all, img_names_list = [], [], [], []

with torch.no_grad():
    for imgs, labels, names in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        y_true.append(labels.item())
        y_pred.append(preds.item())
        probs_all.append(probs.cpu().numpy()[0])
        img_names_list.append(names[0])

# Simpan Hasil Prediksi
df_pred = pd.DataFrame({
    "image_name": img_names_list,
    "y_true": y_true,
    "y_pred": y_pred,
    "true_label_name": [classes[i] for i in y_true],
    "pred_label_name": [classes[i] for i in y_pred]
})
df_pred.to_csv("hasil_prediksi_cnn_simple.csv", index=False)

# Hitung Metrik
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
pixel_accuracy = accuracy # Dalam klasifikasi gambar, pixel acc sering dianggap sama dengan akurasi gambar per gambar

y_true_oh = np.eye(len(classes))[y_true]
mse = mean_squared_error(y_true_oh, probs_all)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Pixel Accuracy", "MSE"],
    "Value": [accuracy, precision, recall, f1, pixel_accuracy, mse]
})

print(metrics_df)
metrics_df.to_csv("evaluation_metrics_cnn_simple.csv", index=False)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN Simple")
plt.savefig("confusion_matrix_cnn.png") # Simpan gambar juga
plt.show()

# =======================
# 8. Save Model
# =======================
save_path = "cnn_simple_cuaca.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': class_to_idx, # Penting disimpan untuk deployment
    'idx_to_class': {v: k for k, v in class_to_idx.items()}
}, save_path)
print(f"✅ Model saved successfully at: {save_path}")