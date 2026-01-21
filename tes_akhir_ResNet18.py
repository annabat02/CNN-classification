import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- KONFIGURASI ---
PATH_MODEL = "resnet18_cuaca.pth" 
PATH_DATASET = "dataset analisa statistik"
IMAGE_SIZE = 150 
BATCH_SIZE = 32

def main():
    # 1. Cek Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Menggunakan device: {device}")

    # 2. Transformasi
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # 3. Load Dataset
    print(f"📂 Memuat data dari: {PATH_DATASET}")
    test_dataset = datasets.ImageFolder(PATH_DATASET, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = test_dataset.classes
    print(f"📂 Kelas ditemukan: {class_names}")

    # 4. MEMBANGUN MODEL & LOAD WEIGHTS (FIXED)
    print("⏳ Sedang membangun arsitektur ResNet18 & memuat bobot...")
    
    # A. Bangun Kerangka Kosong
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 
    
    # B. Load File .pth (KOPER)
    try:
        checkpoint = torch.load(PATH_MODEL, map_location=device)
        
        # C. Logika Cerdas Membuka Koper
        if isinstance(checkpoint, dict):
            # Cek apakah ada kunci 'model_state_dict' (sesuai pesan error kamu tadi)
            if 'model_state_dict' in checkpoint:
                print("✅ Ditemukan kunci 'model_state_dict'. Memuat bobot...")
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                print("✅ Ditemukan kunci 'state_dict'. Memuat bobot...")
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Siapa tahu isinya langsung bobot
                print("⚠️ Tidak ada kunci khusus, mencoba load langsung...")
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
            
        print("🎉 SUKSES! Model berhasil dimuat.")
        
    except Exception as e:
        print(f"\n❌ MASIH ERROR: {e}")
        print("Saran: Cek apakah jumlah kelas di file .pth (misal 5) sama dengan folder dataset.")
        return

    model = model.to(device)
    model.eval() 

    # 5. Prediksi
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

    # 6. Laporan Statistik
    print("\n" + "="*50)
    print("📊 LAPORAN ANALISIS STATISTIK (ResNet18)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya (Aktual)')
    plt.title('Confusion Matrix - ResNet18')
    
    plt.savefig('cm_resnet_final.png')
    print("\n✅ Gambar 'cm_resnet_final.png' berhasil disimpan!")
    plt.show()

if __name__ == "__main__":
    main()