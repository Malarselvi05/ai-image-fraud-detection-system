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
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_food_model.pth")
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

# --------------------------------------------------
# Model Loading
# --------------------------------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

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
# Metadata Scoring (Lightweight)
# --------------------------------------------------
def metadata_score(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is None:
            return 0.05

        exif = {
            ExifTags.TAGS.get(tag): value
            for tag, value in exif_data.items()
            if tag in ExifTags.TAGS
        }

        score = 0.0

        if "Model" not in exif:
            score += 0.03

        if "Software" in exif:
            software = str(exif["Software"]).lower()
            if "stable" in software or "diffusion" in software or "ai" in software:
                score += 0.05

        return min(score, 0.10)

    except:
        return 0.05

# --------------------------------------------------
# Context-Aware Policy
# --------------------------------------------------
def apply_policy(ai_conf, final_conf, upload_type):

    if upload_type == "refund":

        if ai_conf > 0.80:
            return "High Risk - Flag for Review"

        elif ai_conf > 0.40:
            return "Medium Risk - Manual Review"

        else:
            return "Low Risk - Allow"

    elif upload_type == "social":

        if final_conf > 0.85:
            return "High Risk - Flag for Review"
        elif final_conf > 0.70:
            return "Medium Risk - Manual Review"
        else:
            return "Low Risk - Allow"

    else:
        if final_conf > 0.75:
            return "High Risk - Flag for Review"
        elif final_conf > 0.60:
            return "Medium Risk - Manual Review"
        else:
            return "Low Risk - Allow"


# --------------------------------------------------
# Grad-CAM (Called Separately)
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
    handle_b = final_conv.register_backward_hook(backward_hook)

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

    ai_conf = probs[0][0].item()
    real_conf = probs[0][1].item()

    meta_score = metadata_score(image_path)

    final_conf = (0.95 * ai_conf) + (0.05 * meta_score)

    decision = apply_policy(ai_conf, final_conf, upload_type)


    if final_conf > 0.80:
        explanation = "Strong synthetic texture patterns detected."
    elif final_conf > 0.40:
        explanation = "Moderate synthetic artifacts detected."
    else:
        explanation = "No strong synthetic indicators detected."
   

    return {
        "ai_probability": ai_conf,
        "real_probability": real_conf,
        "metadata_score": meta_score,
        "final_score": final_conf,
        "decision": decision,
        "explanation": explanation,
        "inference_time": round(end - start, 4)
    }


if __name__ == "__main__":
    test_image_path = r"D:\Users\Lenova\Desktop\YukTha\dataset\val\ai\7dbef1a7-528f-407e-b368-d3c4e3f16cd3.jpg"
    result = predict_image(test_image_path, upload_type="refund")

    print("\nPrediction Result:")
    for key, value in result.items():
        print(f"{key}: {value}")