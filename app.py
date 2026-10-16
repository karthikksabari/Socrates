"""Real-time demo: upload or capture an image -> predicted class, calibrated confidence, Grad-CAM heatmap, and multi-detector OOD decision.
Run uncertainty_classifier.ipynb first (writes artifacts/model.pt, artifacts/maha.npz, and artifacts/uq.json), then: python app.py"""
import os, json
import numpy as np
import matplotlib.cm as cm
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import gradio as gr
from PIL import Image

cfg = json.load(open("artifacts/uq.json"))
CLASSES, RES, T = cfg["classes"], cfg["res"], cfg["T"]
THR_MSP = cfg["thresholds"]["msp"]
THR_ENERGY = cfg["thresholds"]["energy"]
THR_MAHA = cfg["thresholds"].get("maha", None)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
STD = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)

maha_data = np.load("artifacts/maha.npz") if os.path.exists("artifacts/maha.npz") else None


class Net(nn.Module):                                   # same architecture as in the notebook
    def __init__(self, n_classes):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head(f), f


model = Net(len(CLASSES)).to(dev)
model.load_state_dict(torch.load("artifacts/model.pt", map_location=dev))
model.eval()
conv_layers = nn.Sequential(*list(model.backbone.children())[:-1])


@torch.no_grad()
def predict(img, corruption="None"):
    if img is None:
        return {}, None, "No image uploaded."
    img_rgb = img.convert("RGB").resize((32, 32), Image.BICUBIC)
    x_np = np.array(img_rgb).astype(np.float32) / 255.0

    # Optional real-time corruption stress-test
    if corruption == "Gaussian Noise":
        x_np = np.clip(x_np + np.random.normal(0, 0.12, x_np.shape), 0, 1)
    elif corruption == "Darken (Low Light)":
        x_np = np.clip(x_np * 0.35, 0, 1)
    elif corruption == "Low Resolution (Pixelate)":
        small = Image.fromarray((x_np * 255).astype(np.uint8)).resize((8, 8), Image.BILINEAR)
        x_np = np.array(small.resize((32, 32), Image.NEAREST)).astype(np.float32) / 255.0

    x = torch.from_numpy(x_np).to(dev).permute(2, 0, 1).float().unsqueeze(0)
    if RES != 32:
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x_norm = (x - MEAN) / STD

    logits, feat = model(x_norm)
    p = F.softmax(logits, 1)[0]
    p_cal = F.softmax(logits / T, 1)[0]
    pred_idx = int(p.argmax().item())

    # Compute uncertainty scores
    msp = float(p.max().item())
    energy = float(torch.logsumexp(logits, 1).item())
    entropy = float(-(p * torch.log(p + 1e-12)).sum().item())

    ok_msp = msp >= THR_MSP
    ok_en = energy >= THR_ENERGY
    votes = [ok_msp, ok_en]

    maha_str = ""
    if maha_data is not None and THR_MAHA is not None:
        f_np = feat.cpu().numpy().astype(np.float64)
        mu, P = maha_data["mu"], maha_data["P"]
        d = [float((f_np - m) @ P @ (f_np - m).T) for m in mu]
        maha_score = -min(d)
        ok_maha = maha_score >= THR_MAHA
        votes.append(ok_maha)
        maha_str = f"  |  Mahalanobis {maha_score:.1f} (thr {THR_MAHA:.1f}, {'PASS' if ok_maha else 'REJECT'})"

    n_pass = sum(votes)
    if n_pass == len(votes):
        verdict = f"✅ ACCEPT ({n_pass}/{len(votes)} detectors agree): Familiar In-Distribution Class ({CLASSES[pred_idx]})"
    elif n_pass == 0:
        verdict = f"🛑 REJECT (0/{len(votes)} detectors pass): Unfamiliar / Out-of-Distribution Input!"
    else:
        verdict = f"⚠️ UNCERTAIN ({n_pass}/{len(votes)} detectors pass): Borderline / Corrupted Input"

    # Compute Grad-CAM spatial heatmap overlay
    fmap = conv_layers(x_norm)
    weights = model.head[1].weight[pred_idx]
    cam = F.relu(torch.einsum("bchw,c->bhw", fmap, weights))
    cam_up = F.interpolate(cam.unsqueeze(1), size=(160, 160), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
    heatmap_rgb = (cm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
    base_rgb = np.array(Image.fromarray((x_np * 255).astype(np.uint8)).resize((160, 160), Image.NEAREST))
    overlay = Image.fromarray((0.58 * base_rgb + 0.42 * heatmap_rgb).astype(np.uint8))

    text = (
        f"{verdict}\n"
        f"Top Predicted Known Class: {CLASSES[pred_idx]}  |  Calibrated Confidence (T={T:.2f}): {p_cal.max().item():.3f}  |  Shannon Entropy: {entropy:.3f}\n"
        f"MSP: {msp:.3f} (thr {THR_MSP:.3f}, {'PASS' if ok_msp else 'REJECT'})  |  "
        f"Energy: {energy:.2f} (thr {THR_ENERGY:.2f}, {'PASS' if ok_en else 'REJECT'}){maha_str}"
    )
    return {c: float(v) for c, v in zip(CLASSES, p_cal.cpu().numpy())}, overlay, text


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", sources=["upload", "webcam"], label="Input Image"),
        gr.Radio(["None", "Gaussian Noise", "Darken (Low Light)", "Low Resolution (Pixelate)"], value="None", label="Optional Real-Time Corruption Stress Test")
    ],
    outputs=[
        gr.Label(num_top_classes=len(CLASSES), label="Calibrated Class Probabilities (Temperature Scaled)"),
        gr.Image(type="pil", label="Grad-CAM Spatial Attention Heatmap (160x160)"),
        gr.Textbox(label="Uncertainty Gatekeeper Diagnostics", lines=4)
    ],
    title="The Model That Doesn't Know: Open-Set Classifier & OOD Gatekeeper",
    description="Known classes: " + ", ".join(CLASSES) + ". Upload any known object or unfamiliar OOD object (e.g., deer, frog, horse, truck, or random photo) to inspect calibrated probabilities, multi-detector OOD scores, and Grad-CAM attention."
)

if __name__ == "__main__":
    demo.launch()
