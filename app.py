import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# clone this
#git clone https://github.com/ggjy/CMT.pytorch.git

# --- Make sure Python can see the local repo: ./CMT.pytorch ---
REPO_DIR = Path(__file__).parent / "CMT.pytorch"
if not REPO_DIR.exists():
    st.error(f"Required repo folder not found: {REPO_DIR}. "
             f"Place the CMT.pytorch folder next to this app.py.")
    st.stop()

sys.path.append(str(REPO_DIR))

try:
    from cmt import cmt_ti  # same constructor used during training
except Exception as e:
    st.error("Could not import 'cmt_ti' from the CMT.pytorch repo. "
             "Verify the repo contents and package structure.")
    st.exception(e)
    st.stop()

# --- App constants ---
CLASSES = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Tooth Discoloration", "Ulcers"]
CKPT_PATH = Path(__file__).parent / "best_cmt_ti_dental_classifier1.pth" 

# --- Image transforms (match training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Model loader (cached) ---
@st.cache_resource
def load_model():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}. "
            "Put your trained weights there or update CKPT_PATH."
        )

    # Build the same architecture used during training
    model = cmt_ti(pretrained=False, num_classes=len(CLASSES))

    # Load backbone / full state dict; allow classifier mismatch if backbone-only
    state = torch.load(CKPT_PATH, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()  # inference mode on CPU by default

    return model, missing, unexpected

# --- UI ---
st.title("🦷 Dental Disease Classifier")

# Load model (and show any state_dict warnings to help debugging)
try:
    model, missing, unexpected = load_model()
    if missing or unexpected:
        with st.expander("Model load details (debug)"):
            st.write("Missing keys:", missing)
            st.write("Unexpected keys:", unexpected)
except Exception as e:
    st.error("Failed to load the model.")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader("Upload a dental image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess & run inference
    input_tensor = transform(image).unsqueeze(0)  # type: ignore # [1, C, H, W]
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top_prob, top_class = torch.max(probs, dim=0)

    st.subheader("Prediction")
    st.write(f"**{CLASSES[int(top_class)]}** ({top_prob.item():.2%})")

    st.subheader("Class Probabilities")
    for i, cls in enumerate(CLASSES):
        st.write(f"{cls}: {probs[i].item():.2%}")
