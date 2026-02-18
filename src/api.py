import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import numpy as np
import cv2
import time
import os

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_food_model_v4.pth")
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

# --------------------------------------------------
# Model Loading
# --------------------------------------------------
model = models.mobilenet_v2(weights=None)

# IMPORTANT: Must match training architecture EXACTLY
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✅ mobilenet_food_model_v4 loaded successfully")
# --------------------------------------------------
# Transform
# --------------------------------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------------------------------
# Metadata Scoring
# --------------------------------------------------
def metadata_info(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is None:
            return False

        exif = {
            ExifTags.TAGS.get(tag): value
            for tag, value in exif_data.items()
            if tag in ExifTags.TAGS
        }

        if "Model" in exif:
            return True

        return False

    except:
        return False



# --------------------------------------------------
# Context-Aware Policy
# --------------------------------------------------
def apply_policy(ai_conf, exif_present, upload_type):

    # -----------------------------
    # REFUND FLOW (More Strict)
    # -----------------------------
    if upload_type == "refund":

        if ai_conf > 0.95:
            # If camera metadata exists, reduce severity
            if exif_present:
                return "Medium Risk - Manual Review"
            else:
                return "High Risk - Flag for Review"

        elif ai_conf > 0.80:
            return "Medium Risk - Manual Review"

        else:
            return "Low Risk - Allow"


    # -----------------------------
    # SOCIAL FLOW (Balanced)
    # -----------------------------
    elif upload_type == "social":

        if ai_conf > 0.90:
            if exif_present:
                return "Medium Risk - Manual Review"
            else:
                return "High Risk - Flag for Review"

        elif ai_conf > 0.75:
            return "Medium Risk - Manual Review"

        else:
            return "Low Risk - Allow"


    # -----------------------------
    # DEFAULT FLOW
    # -----------------------------
    else:

        if ai_conf > 0.90:
            if exif_present:
                return "Medium Risk - Manual Review"
            else:
                return "High Risk - Flag for Review"

        elif ai_conf > 0.70:
            return "Medium Risk - Manual Review"

        else:
            return "Low Risk - Allow"

# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------
def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    final_conv = model.features[-1]

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = final_conv.register_forward_hook(forward_hook)
    handle_b = final_conv.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1)

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))

    if np.max(cam) != 0:
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + original * 0.6

    heatmap_path = os.path.join(HEATMAP_DIR, os.path.basename(image_path))
    cv2.imwrite(heatmap_path, overlay)

    handle_f.remove()
    handle_b.remove()

    return heatmap_path

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_image(image_path, upload_type="refund"):

    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
    end = time.time()

    # ⚠️ IMPORTANT: Confirm class order from training
    # If {'ai': 0, 'real': 1}
    ai_conf = probs[0][0].item()
    real_conf = probs[0][1].item()

    # Metadata presence (True/False)
    exif_present = metadata_info(image_path)

    # For explanation only
    final_conf = ai_conf

    decision = apply_policy(ai_conf, exif_present, upload_type)

    if ai_conf > 0.80:
        explanation = "Strong synthetic texture patterns detected."
    elif ai_conf > 0.40:
        explanation = "Moderate synthetic artifacts detected."
    else:
        explanation = "No strong synthetic indicators detected."

    print("Raw logits:", output)
    print("Softmax probs:", probs)
    print("EXIF present:", exif_present)

    return {
        "ai_probability": round(ai_conf, 4),
        "real_probability": round(real_conf, 4),
        "metadata_present": exif_present,
        "final_score": round(final_conf, 4),
        "decision": decision,
        "explanation": explanation,
        "inference_time": round(end - start, 4)
    }