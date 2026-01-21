import os
import imagehash
from PIL import Image
import numpy as np

# --- KONFIGURASI PATH SESUAI PERMINTAAN ---
# Menggunakan path relatif. Pastikan script ini dijalankan di luar folder 'data'
PATH_DATASET_TRAIN = "data"  
PATH_DATASET_TEST_BARU = "dataset analisa statistik"

def get_all_image_paths(directory):
    """Mengambil semua path gambar dalam direktori termasuk subfolder."""
    image_paths = []
    # Ekstensi yang didukung
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Cek apakah folder ada
    if not os.path.exists(directory):
        print(f"❌ Error: Folder '{directory}' tidak ditemukan!")
        return []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def check_for_leakage(train_dir, test_dir, threshold=5):
    print(f"🚀 Memulai pemeriksaan integritas data...")
    print(f"📂 Dataset Train (Utama) : '{train_dir}'")
    print(f"📂 Dataset Baru (Uji)    : '{test_dir}'")
    print("-" * 50)

    # 1. Ambil list file
    train_paths = get_all_image_paths(train_dir)
    test_paths = get_all_image_paths(test_dir)

    if not train_paths:
        print("⚠️ Tidak ada gambar di folder dataset utama. Cek path-nya lagi.")
        return
    if not test_paths:
        print("⚠️ Tidak ada gambar di folder dataset baru. Cek path-nya lagi.")
        return

    # 2. Bangun Database Hash dari Dataset Training
    print(f"1️⃣  Membuat 'sidik jari' (hash) dari {len(train_paths)} gambar training...")
    train_hashes = {}
    
    for path in train_paths:
        try:
            with Image.open(path) as img:
                # phash tahan terhadap resize/kompresi ringan
                h = imagehash.phash(img)
                train_hashes[h] = path
        except Exception as e:
            print(f"   ⚠️ Gagal membaca {os.path.basename(path)}")

    print(f"   ✅ Selesai hashing dataset utama.\n")

    # 3. Periksa Dataset Baru terhadap Database Training
    print(f"2️⃣  Memeriksa {len(test_paths)} gambar baru terhadap dataset utama...")
    leak_count = 0
    
    for path in test_paths:
        is_leak = False
        try:
            with Image.open(path) as img:
                test_hash = imagehash.phash(img)
                
                # Bandingkan hash gambar baru dengan SEMUA hash di dataset utama
                for train_h, train_p in train_hashes.items():
                    # Menghitung Hamming Distance (selisih bit)
                    dist = test_hash - train_h
                    
                    if dist <= threshold:
                        print(f"❌ KEBOCORAN DITEMUKAN!")
                        print(f"   Baru  : {path}")
                        print(f"   Lama  : {train_p}")
                        print(f"   Jarak : {dist} (Sangat mirip/sama)")
                        is_leak = True
                        leak_count += 1
                        break # Stop loop, pindah ke gambar berikutnya
                        
        except Exception as e:
            print(f"   ⚠️ Error pada file {os.path.basename(path)}: {e}")

    # 4. Kesimpulan
    print("-" * 50)
    print("📊 LAPORAN AKHIR")
    print(f"Total Gambar Baru Diperiksa : {len(test_paths)}")
    print(f"Total Kebocoran (Leakage)   : {leak_count}")
    
    if leak_count == 0:
        print("\n✅ STATUS: AMAN & VALID")
        print("Dataset 'dataset analisa statistik' benar-benar BARU.")
        print("Tidak ada tumpang tindih visual dengan dataset 'data'.")
    else:
        print("\n⚠️ STATUS: PERLU PERBAIKAN")
        print(f"Ada {leak_count} gambar yang terdeteksi duplikat/mirip.")
        print("Silakan hapus gambar yang terdeteksi di atas dari folder 'dataset analisa statistik'.")

# Jalankan fungsi
if __name__ == "__main__":
    check_for_leakage(PATH_DATASET_TRAIN, PATH_DATASET_TEST_BARU)