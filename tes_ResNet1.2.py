import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# ============================================================
# KONFIGURASI (HARUS KONSISTEN DENGAN TRAINING)
# ============================================================
PATH_MODEL = "resnet18_cuaca.pth"
PATH_DATASET = "dataset analisa statistik"
IMAGE_SIZE = 224          # WAJIB 224 (ResNet)
BATCH_SIZE = 32

def main():
    # ============================================================
    # 1. DEVICE
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Menggunakan device: {device}")

    # ============================================================
    # 2. TRANSFORMASI (SAMA DENGAN transform_test DI TRAINING)
    # ============================================================
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f"📂 Memuat dataset uji eksternal dari: {PATH_DATASET}")
    test_dataset = datasets.ImageFolder(PATH_DATASET, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📂 Kelas ditemukan: {class_names}")

    # ============================================================
    # 3. MEMBANGUN MODEL & LOAD BOBOT HASIL TRAINING
    # ============================================================
    print("⏳ Membangun arsitektur ResNet18 & memuat bobot training...")

    # Bangun arsitektur (tanpa download pretrained lagi)
    model = models.resnet18(weights=None)

    # Ganti FC layer sesuai jumlah kelas
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load bobot hasil training
    checkpoint = torch.load(PATH_MODEL, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("✅ Bobot model berhasil dimuat")

    model = model.to(device)
    model.eval()

    # ============================================================
    # 4. INFERENSI / PREDIKSI
    # ============================================================
    y_true = []
    y_pred = []

    print("⚡ Melakukan prediksi pada dataset uji eksternal...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ============================================================
    # 5. LAPORAN ANALISIS STATISTIK
    # ============================================================
    print("\n" + "=" * 55)
    print("📊 LAPORAN ANALISIS STATISTIK (ResNet18 - Transfer Learning)")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ============================================================
    # 6. CONFUSION MATRIX
    # ============================================================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya')
    plt.title('Confusion Matrix - ResNet18 (Uji Eksternal)')
    plt.tight_layout()
    plt.savefig('cm_resnet18_uji_eksternal.png')
    plt.show()

    print("✅ Confusion Matrix berhasil disimpan")

if __name__ == "__main__":
    main()
