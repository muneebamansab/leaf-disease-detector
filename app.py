import streamlit as st
import torch, joblib, numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTModel

st.set_page_config(page_title="LeafGuard AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%);}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: #14532d !important;}
[data-testid="stSidebar"] {background: #bbf7d0;}
[data-testid="stHeader"] {background: transparent;}
.hero {background: linear-gradient(135deg,#14532d 0%,#16a34a 55%,#86efac 100%);
       padding: 2.2rem; border-radius: 22px; margin-bottom: 1.5rem;
       box-shadow: 0 8px 24px rgba(22,163,74,.3);}
.hero h1, .hero p {color: #ffffff !important; margin: 0;}
.hero h1 {font-size: 2.8rem;}
.hero p {margin-top: .5rem; font-size: 1.1rem;}
.card {background: #ffffff; border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
       box-shadow: 0 4px 18px rgba(0,0,0,.08); border-left: 6px solid #16a34a;}
.card.bad {border-left-color: #dc2626;}
.card.warn {border-left-color: #f59e0b;}
.badge {display: inline-block; padding: .25rem .8rem; border-radius: 999px; font-weight: 600;
        font-size: .85rem; background: #16a34a; color: #ffffff !important;}
.badge.bad {background: #dc2626;}
.badge.warn {background: #f59e0b;}
.bar-wrap {background: #bbf7d0; border-radius: 999px; height: 12px; margin: .25rem 0 .8rem;}
.bar {background: linear-gradient(90deg,#16a34a,#4ade80); height: 12px; border-radius: 999px;}
.crop {background: #ffffff; border-radius: 14px; padding: .8rem; text-align: center;
       box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-bottom: .8rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

CROPS = {"Apple": "🍎", "Blueberry": "🫐", "Cherry": "🍒", "Corn": "🌽", "Grape": "🍇",
         "Orange": "🍊", "Peach": "🍑", "Pepper": "🫑", "Potato": "🥔", "Raspberry": "🍓",
         "Soybean": "🌱", "Squash": "🎃", "Strawberry": "🍓", "Tomato": "🍅"}

def tip_for(condition):
    c = condition.lower()
    if "healthy" in c:
        return "The leaf looks healthy. Keep up regular watering and good airflow, and keep checking for spots."
    if "virus" in c:
        return "Viral disease with no cure. Remove infected plants, control insects like whiteflies and aphids, and use resistant varieties."
    if "haunglongbing" in c:
        return "Citrus greening is spread by psyllids and has no cure. Remove infected trees, control psyllids, and inform your local agriculture office."
    if "bacterial" in c:
        return "Bacterial disease. Avoid overhead watering, remove infected leaves, and ask a local expert about copper-based sprays."
    if "mite" in c:
        return "Spider mites. Rinse leaves with water, use insecticidal soap, and avoid drought stress."
    return "Likely a fungal disease. Remove infected leaves, improve airflow, avoid wetting leaves, and ask a local expert about a suitable fungicide."

def pretty(label):
    crop, cond = label.split("___")
    return crop.replace("_", " ").strip(), cond.replace("_", " ").strip()

@st.cache_resource(show_spinner="Loading AI model (first run takes a minute)...")
def load_all():
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224").eval()
    model = joblib.load("leaf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    classes = list(joblib.load("classes.pkl"))
    return vit, model, scaler, classes

vit, model, scaler, classes = load_all()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def predict(img):
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        emb = vit(x).last_hidden_state[:, 0].numpy()
    probs = model.predict_proba(scaler.transform(emb))[0]
    idx = np.argsort(probs)[::-1][:3]
    return [(classes[i], float(probs[i])) for i in idx]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌿 LeafGuard AI")
    st.caption("Plant leaf disease detection")
    st.metric("Validation accuracy", "99.2%")
    st.metric("Classes", "38")
    st.metric("Crops", "14")
    st.markdown("---")
    st.markdown("**Model:** Vision Transformer (ViT) features + Logistic Regression")
    st.markdown("**Dataset:** PlantVillage (~54k images)")
    st.markdown("---")
    st.caption("COMSATS University Islamabad, Lahore Campus  \nAIC354 Semester Project")

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <h1>🌿 LeafGuard AI</h1>
  <p>Upload a photo of a plant leaf and get an instant disease diagnosis, powered by a Vision Transformer.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Detect disease", "ℹ️ How it works", "🌱 Supported crops"])

# ---------- Tab 1: Detect ----------
with tab1:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        file = st.file_uploader("Upload a leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Your leaf", use_container_width=True)
        else:
            st.info("Upload a clear photo of a single leaf on a plain background for the best results.")

    with right:
        if file:
            with st.spinner("Analyzing leaf..."):
                top3 = predict(img)
            label, conf = top3[0]
            crop, cond = pretty(label)
            healthy = "healthy" in cond.lower()

            if conf < 0.60:
                st.markdown(f"""
                <div class="card warn"><span class="badge warn">Not sure</span>
                <h3>Low confidence ({conf*100:.0f}%)</h3>
                <p>This leaf may not be one of the supported plants, or the photo is unclear.
                Best guess: <b>{crop} - {cond}</b>. Try a clearer photo.</p></div>""",
                unsafe_allow_html=True)
            else:
                kind = "" if healthy else "bad"
                badge = "Healthy" if healthy else "Disease detected"
                st.markdown(f"""
                <div class="card {kind}"><span class="badge {kind}">{badge}</span>
                <h2 style="margin:.5rem 0 0">{CROPS.get(crop.split()[0], "🌿")} {crop}</h2>
                <h3 style="margin:.1rem 0 .5rem">{cond}</h3>
                <p>Confidence: <b>{conf*100:.1f}%</b></p></div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="card"><b>💡 What to do</b><br>{tip_for(cond)}</div>',
                            unsafe_allow_html=True)

            st.markdown("#### Top 3 predictions")
            for lab, p in top3:
                c, d = pretty(lab)
                st.markdown(f"**{c} - {d}** ({p*100:.1f}%)")
                st.markdown(f'<div class="bar-wrap"><div class="bar" style="width:{p*100:.1f}%"></div></div>',
                            unsafe_allow_html=True)
            st.caption("Educational tool only. Please confirm with an agricultural expert before treating crops.")

# ---------- Tab 2: How it works ----------
with tab2:
    st.markdown("""
<div class="card"><b>1. Upload</b><br>Your leaf photo is resized to 224x224 pixels.</div>
<div class="card"><b>2. Feature extraction</b><br>A pretrained Vision Transformer (ViT) turns the image into 768 numbers that describe it.</div>
<div class="card"><b>3. Classification</b><br>A Logistic Regression classifier, trained on ~43k PlantVillage images, picks the most likely of 38 classes.</div>
<div class="card"><b>4. Result</b><br>You see the crop, the condition, the confidence, and a short care tip.</div>
""", unsafe_allow_html=True)
    st.subheader("Limitations")
    st.markdown("""
- Only the 14 crops and 38 classes in PlantVillage are supported. Other plants will be forced into one of them.
- Training photos are clean lab images, so accuracy may drop on field photos with messy backgrounds.
- Blueberry, raspberry, and soybean have only healthy samples, so diseases can't be detected for them.
- Some classes have far fewer images than others (class imbalance).
""")

# ---------- Tab 3: Crops ----------
with tab3:
    cols = st.columns(4)
    for i, (name, emoji) in enumerate(CROPS.items()):
        with cols[i % 4]:
            st.markdown(f'<div class="crop">{emoji}<br>{name}</div>', unsafe_allow_html=True)