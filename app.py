"""
app.py - Gradio arayüzü (Deployment)

Kullanıcı bir kıyafet/aksesuar fotoğrafı yükler, model tahmini ve olasılıkları gösterilir.
Çalıştırmak için:  python app.py   ->  tarayıcıda http://127.0.0.1:7860
Not: Önce notebook çalıştırılıp outputs/fashion_cnn.pth oluşturulmalıdır.
"""
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

CLASSES = ["Tişört/Üst", "Pantolon", "Kazak", "Elbise", "Mont/Ceket",
           "Sandalet", "Gömlek", "Spor ayakkabı", "Çanta", "Bot"]


# Notebook'taki modelin aynısı
class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32 x 14 x 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 64 x 7 x 7
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Notebook'taki ön işleme fonksiyonunun aynısı
def preprocess_photo(bgr):
    """Telefon fotoğrafını Fashion-MNIST formatına (28x28, siyah zemin) çevirir."""
    if bgr.dtype == np.uint16:                       # 16-bit PNG ise 8-bit'e çevir
        bgr = (bgr >> 8).astype(np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    # 1) küçült ve Lab renk uzayına çevir
    h, w = bgr.shape[:2]
    s = 600 / max(h, w)
    small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 2) zemin rengi = kenar piksellerinin medyanı, her pikselin zemine uzaklığı
    H, W = lab.shape[:2]
    b = max(4, int(0.04 * min(H, W)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    diff = lab - np.median(border, axis=0)
    dist = np.sqrt(diff[..., 0]**2 + 4 * diff[..., 1]**2 + 4 * diff[..., 2]**2)
    dist = cv2.GaussianBlur(dist, (9, 9), 0)
    d8 = np.clip(dist / dist.max() * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # GrabCut ile maskeyi iyileştir
    gc = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[:b] = gc[-b:] = cv2.GC_BGD
    gc[:, :b] = gc[:, -b:] = cv2.GC_BGD
    cv2.setRNGSeed(42)                               # GrabCut her seferinde aynı sonucu versin
    cv2.grabCut(blur, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # en büyük parça(lar)ı tut, içini doldur
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * areas.max()]
    mask = np.isin(labels, keep).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnts, -1, 255, cv2.FILLED)

    # 3) gri ton, ters çevirme, zemin = 0, ürün kontrastını 80-255 aralığına ger
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    inv = (255 - gray).astype(np.float32)
    m = mask > 0
    lo, hi = np.percentile(inv[m], [2, 98])
    item = np.clip((inv - lo) / max(hi - lo, 1) * 175 + 80, 0, 255)
    item[~m] = 0

    # 4) kırp, uzun kenar 28 piksel olacak şekilde küçült, ortala
    ys, xs = np.nonzero(mask)
    crop = item[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = 28 / max(crop.shape)
    nh, nw = max(1, round(crop.shape[0] * sc)), max(1, round(crop.shape[1] * sc))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.float32)
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = crop
    return out.astype(np.uint8), small, mask


model = FashionCNN()
model.load_state_dict(torch.load("outputs/fashion_cnn.pth", map_location="cpu"))
model.eval()


def predict(image):
    if image is None:
        return None, None, "Lütfen bir fotoğraf yükleyin."
    if image.ndim == 3 and image.shape[2] == 4:          # PNG'de saydamlık kanalı varsa at
        image = image[:, :, :3]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)       # Gradio RGB verir, OpenCV BGR kullanır
    try:
        x28, _, _ = preprocess_photo(bgr)
    except Exception as e:
        return None, None, f"Ürün bulunamadı: {e}"
    x = torch.tensor(x28 / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].numpy()
    ms = (time.perf_counter() - t0) * 1000
    labels = {CLASSES[i]: float(probs[i]) for i in range(10)}
    big = cv2.resize(x28, (224, 224), interpolation=cv2.INTER_NEAREST)
    info = f"Tahmin: {CLASSES[int(probs.argmax())]}  (güven %{probs.max()*100:.1f}, süre {ms:.2f} ms)"
    return labels, big, info


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Fotoğraf yükle"),
    outputs=[gr.Label(num_top_classes=3, label="Tahmin"),
             gr.Image(label="Modele giren görüntü (28x28)"),
             gr.Textbox(label="Sonuç")],
    title="Fashion-MNIST Kıyafet Sınıflandırma (CNN)",
    description="Düz bir zemine serilmiş TEK bir ürünün fotoğrafını yükleyin: tişört, pantolon, kazak, elbise, mont, "
                "sandalet, gömlek, spor ayakkabı, çanta veya bot.",
    examples=[["gercek_fotograflar/8_canta.jpg"], ["gercek_fotograflar/1_kot_pantolon.jpg"]],
)

if __name__ == "__main__":
    demo.launch()
