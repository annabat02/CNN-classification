import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights 
from PIL import Image
import os

# =====================
# CONFIG
# =====================
# Urutan ABJAD: cloudy -> foggy -> rainy -> snowy -> sunny
CLASS_NAMES = ["cloudy", "foggy", "rainy", "snowy", "sunny"]

# =====================
# DEFINISI MODEL
# =====================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_resnet18_structure(num_classes):
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# =====================
# STREAMLIT UI SETUP
# =====================
st.set_page_config(page_title="Prediksi Cuaca", page_icon="🌤️")

# CSS: Mengurangi padding atas/bawah agar muat screenshot
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 { margin-bottom: 0rem; }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem; /* Mengurangi jarak antar elemen */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌤️ Klasifikasi Citra Cuaca")
st.write("Upload gambar langit/cuaca untuk diprediksi.")

# Sidebar
st.sidebar.header("Konfigurasi Model")
model_choice = st.sidebar.selectbox("Pilih Arsitektur Model:", ["CNN Simple", "ResNet18 (Transfer Learning)"])

uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg","png","jpeg"])

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model_logic(choice):
    device = torch.device("cpu") 
    
    if choice == "CNN Simple":
        model = SimpleCNN(len(CLASS_NAMES))
        path = "cnn_simple_cuaca.pth" 
    else:  # ResNet18
        model = load_resnet18_structure(len(CLASS_NAMES))
        path = "resnet18_cuaca.pth"   

    if not os.path.exists(path):
        st.error(f"⚠️ File model '{path}' tidak ditemukan!")
        return None

    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) 
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model_logic(model_choice)

# =====================
# TRANSFORMASI
# =====================
def get_transform(choice):
    if choice == "CNN Simple":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    else:  # ResNet18
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

if model is not None:
    transform = get_transform(model_choice)

# =====================
# PROSES PREDIKSI & TAMPILAN
# =====================
if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Gambar Input", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_label = CLASS_NAMES[pred_idx]
    pred_score = probs[pred_idx].item()

    with col2:
        emoji_dict = {"cloudy": "☁️", "foggy": "🌫️", "rainy": "🌧️", "sunny": "☀️", "snowy": "❄️"}
        emoji = emoji_dict.get(pred_label, "🌤️")
        
        # 1. Hasil Utama (Compact Box)
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724; border: 1px solid #c3e6cb; margin-bottom: 10px;">
            <h3 style="margin:0; padding:0;">{emoji} {pred_label.upper()}</h3>
            <p style="margin:0; padding:0; font-weight:bold; font-size:16px;">
                Confidence: {pred_score*100:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 2. Garis Tipis
        st.markdown("""<hr style="margin: 5px 0px; border: none; border-top: 1px solid #ccc;" />""", unsafe_allow_html=True)

        # 3. Detail Probabilitas (Ada Angkanya)
        st.markdown("**Detail Probabilitas:**")
        
        for i, cls in enumerate(CLASS_NAMES):
            score = probs[i].item()
            percentage = f"{score*100:.2f}%"
            
            # Layout: Kolom Teks (Nama + %) di kiri, Bar di kanan
            c_text, c_bar = st.columns([1, 1.5]) 
            
            with c_text:
                if i == pred_idx:
                    # Tampilkan Nama Kelas DAN Persentase (Bold untuk prediksi terpilih)
                    st.markdown(f"**👉 {cls}: {percentage}**")
                else:
                    # Tampilkan Nama Kelas DAN Persentase
                    st.markdown(f"{cls}: {percentage}")
            
            with c_bar:
                st.progress(score)