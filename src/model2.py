import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import os
from lime import lime_image
from src.explainability_engine import generate_gradcam

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# --------------------------------------------------
# Model Loading
# --------------------------------------------------
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 2)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("✅ EfficientNet B0 loaded successfully")

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
# LIME Structured (CPU Optimized)
# --------------------------------------------------
def generate_lime_structured(image_path):

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image)

    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        batch = torch.stack([
            val_transform(Image.fromarray(img.astype("uint8")))
            for img in images
        ]).to(device)

        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.detach().cpu().numpy()

    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=1,
        num_samples=50
    )

    class_id = explanation.top_labels[0]
    local_exp = explanation.local_exp[class_id]

    positive_count = 0
    negative_count = 0
    strongest_weight = 0

    for region_id, weight in local_exp[:5]:
        if weight > 0:
            positive_count += 1
        else:
            negative_count += 1

        strongest_weight = max(strongest_weight, abs(weight))

    if positive_count > negative_count:
        interpretation = (
            "Regions show smooth texture patterns often linked with AI-generated content."
        )
    elif negative_count > positive_count:
        interpretation = (
            "Regions show natural irregular patterns suggesting real-world capture."
        )
    else:
        interpretation = (
            "Mixed regional characteristics detected."
        )

    explanation_strength = min(strongest_weight * 3, 1.0)

    return {
        "lime_summary": {
            "ai_supporting_regions": positive_count,
            "real_supporting_regions": negative_count,
            "interpretation": interpretation,
            "explanation_strength": round(explanation_strength, 4)
        }
    }

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
def predict_image(image_path, risk_mode="moderate_scrutiny"):

    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    start = time.time()

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)

    end = time.time()

    raw_ai_conf = probs[0][0].item()
    raw_real_conf = probs[0][1].item()

    # -----------------------------
    # Confidence Calibration
    # -----------------------------
    calibrated_ai = 1 / (1 + np.exp(-3.5 * (raw_ai_conf - 0.45)))
    calibrated_real = 1 / (1 + np.exp(-3.5 * (raw_real_conf - 0.45)))

    total = calibrated_ai + calibrated_real
    ai_conf = float(calibrated_ai / total)
    real_conf = float(calibrated_real / total)

    # --------------------------------------------------
    # YOUR CUSTOM RISK RULE
    # --------------------------------------------------
    if ai_conf > 0.80:
        label = "AI-Generated"
        confidence_level = "HIGH"
        decision = "High Risk - Upload Blocked"
        risk_tier = "HIGH"

    elif ai_conf > 0.40:
        label = "AI-Generated"
        confidence_level = "MEDIUM"
        decision = "Medium Risk - Manual Review Required"
        risk_tier = "MEDIUM"

    else:
        label = "Real"
        confidence_level = "LOW"
        decision = "Low Risk - Upload Approved"
        risk_tier = "LOW"

    result = {
        "prediction": {
            "label": label,
            "confidence_level": confidence_level,
            "raw_ai_probability": round(raw_ai_conf, 4),
            "raw_real_probability": round(raw_real_conf, 4),
            "calibrated_ai_probability": round(ai_conf, 4),
            "calibrated_real_probability": round(real_conf, 4),
            "decision": decision,
            "risk_tier": risk_tier
        },
        "inference_time": round(end - start, 4)
    }

    # Only generate Grad-CAM if Medium or High
   # if risk_tier in ["HIGH", "MEDIUM"]:
    #    gradcam_data = generate_gradcam(
     #       image_path,
      #      model,
        #    val_transform,
         #   device
        #)
      #  result["visualization"] = gradcam_data
  #  else:
    result["visualization"] = None

    return result


# --------------------------------------------------
# Explainability
# --------------------------------------------------
def generate_full_explainability(image_path, ai_conf, real_conf):

    gradcam_data = generate_gradcam(
        image_path,
        model,
        val_transform,
        device
    )

    lime_data = generate_lime_structured(image_path)

    explanation_strength = lime_data["lime_summary"]["explanation_strength"]

    confidence_margin = abs(ai_conf - real_conf)

    reliability_score = round(
        (confidence_margin * 0.6) +
        (explanation_strength * 0.4),
        4
    )

    label = "AI-Generated" if ai_conf > real_conf else "Real"
    top_prob = max(ai_conf, real_conf)

    summary = (
        f"Image classified as {label} with probability {top_prob:.4f}. "
        f"Overall reliability score: {reliability_score:.2f}."
    )

    return {
        "prediction": {
            "label": label,
            "ai_probability": round(ai_conf, 4),
            "real_probability": round(real_conf, 4),
        },
        "explainability": {
            "grad_cam": gradcam_data,
            "lime": lime_data,
            "reliability_score": reliability_score,
            "summary": summary
        }
    }


# --------------------------------------------------
# Local Test
# --------------------------------------------------
if __name__ == "__main__":

    test_image_path = r"D:\Users\Lenova\Downloads\test.jpg"
    result = predict_image(test_image_path, "moderate_scrutiny")

    print("\n🔍 Structured Prediction Result:\n")
    print(result)
