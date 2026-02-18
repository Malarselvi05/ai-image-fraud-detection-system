import os
import cv2
import base64
import numpy as np
from PIL import Image
from uuid import uuid4
from io import BytesIO


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)


# --------------------------------------------------
# Structured Grad-CAM
# --------------------------------------------------
def generate_gradcam(image_path, model, transform, device):

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    final_conv = model.features[-1]
    handle_f = final_conv.register_forward_hook(forward_hook)
    handle_b = final_conv.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1)

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, image.size)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    original = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    filename = f"{uuid4()}.png"
    save_path = os.path.join(HEATMAP_DIR, filename)
    Image.fromarray(overlay).save(save_path)

    handle_f.remove()
    handle_b.remove()

    return {
        "heatmap_path": save_path,
        "heatmap_url": f"/heatmaps/{filename}"
    }