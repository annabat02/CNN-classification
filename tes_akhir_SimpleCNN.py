import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- KONFIGURASI KHUSUS SIMPLE CNN ---
PATH_MODEL = "cnn_simple_cuaca.pth" 
PATH_DATASET = "dataset analisa statistik"
IMAGE_SIZE = 128  # <--- WAJIB 128 (Sesuai training code kamu)
BATCH_SIZE = 32

# ==========================================
# 1. DEFINISI ARSITEKTUR MODEL (WAJIB ADA)
# ==========================================
# Kita harus mendefinisikan ulang kelas SimpleCNN agar Python tahu bentuk "wadahnya"
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Perhitungan input linear sesuai ukuran gambar 128x128
        # 128 -> 64 -> 32 -> 16 (Setelah 3x MaxPool)
        # Feature map akhir: 64 channel x 16 x 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 2. MAIN PROGRAM
# ==========================================
def main():
    # Cek Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Menggunakan device: {device}")

    # Transformasi (Sesuai Training: Resize 128 & ToTensor)
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load Dataset
    print(f"📂 Memuat data dari: {PATH_DATASET}")
    if not os.path.exists(PATH_DATASET):
        print("❌ Error: Folder dataset tidak ditemukan!")
        return

    test_dataset = datasets.ImageFolder(PATH_DATASET, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = test_dataset.classes
    print(f"📂 Kelas ditemukan: {class_names}")

    # --- LOAD MODEL SIMPLE CNN ---
    print("⏳ Sedang membangun arsitektur Simple CNN & memuat bobot...")
    
    # A. Inisialisasi Model Sesuai Jumlah Kelas
    model = SimpleCNN(num_classes=len(class_names))
    
    # B. Load Bobot dari file .pth
    try:
        checkpoint = torch.load(PATH_MODEL, map_location=device)
        
        # C. Logika Ekstrak Bobot
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("✅ Ditemukan kunci 'model_state_dict'. Memuat bobot...")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("⚠️ Tidak ada kunci khusus, mencoba load langsung...")
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
            
        print("🎉 SUKSES! Model Simple CNN berhasil dimuat.")
        
    except Exception as e:
        print(f"\n❌ ERROR SAAT LOAD MODEL: {e}")
        return

    model = model.to(device)
    model.eval() 

    # --- PREDIKSI ---
    y_true = []
    y_pred = []
    
    print("⚡ Melakukan prediksi...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # --- LAPORAN STATISTIK ---
    print("\n" + "="*50)
    print("📊 LAPORAN ANALISIS STATISTIK (Simple CNN)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya (Aktual)')
    plt.title('Confusion Matrix - Simple CNN')
    
    nama_file_cm = 'cm_simple_cnn_final.png'
    plt.savefig(nama_file_cm)
    print(f"\n✅ Gambar '{nama_file_cm}' berhasil disimpan!")
    plt.show()

if __name__ == "__main__":
    main()