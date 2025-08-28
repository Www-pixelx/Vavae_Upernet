import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from train_v4 import *
# 复用你代码里的 CONFIG 与 build_segmentation_model()
# from your_module import CONFIG, build_segmentation_model


device = "cuda" if torch.cuda.is_available() else "cpu"

# 和验证一致的预处理
val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(),           # 默认缩放到 [0,1]，不减均值不除方差
    ToTensorV2()
])

def load_model(ckpt_path: str, device: str):
    model = build_segmentation_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model

model = load_model("/root/vavae-upernet/best_segmentation_model_v4.pth", device)

def voc_palette():
    # 生成 PASCAL VOC 256x3 调色板
    palette = [0] * 256 * 3
    for j in range(256):
        lab = j
        r = g = b = 0
        for i in range(8):
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            lab >>= 3
        palette[j*3+0] = r
        palette[j*3+1] = g
        palette[j*3+2] = b
    return palette

def preprocess_pil(img_pil: Image.Image):
    img_np = np.array(img_pil.convert("RGB"))
    h0, w0 = img_np.shape[:2]
    sample = val_transform(image=img_np)
    tensor = sample["image"].unsqueeze(0)  # [1,3,H,W]
    return tensor, (h0, w0)

@torch.inference_mode()
def predict_logits(model, x):
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        return model(x.to(device))  # [1,C,h,w], h=w=CONFIG["image_size"]

def predict_mask(model, img_pil: Image.Image, keep_size=True):
    x, (h0, w0) = preprocess_pil(img_pil)
    logits = predict_logits(model, x)
    if keep_size and logits.shape[-2:] != (h0, w0):
        logits = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=True)
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # [H,W]
    return pred

def save_color_mask(mask: np.ndarray, save_path: str):
    m = Image.fromarray(mask, mode="P")
    m.putpalette(voc_palette())
    m.save(save_path)

def overlay_mask(img_pil: Image.Image, mask: np.ndarray, alpha=0.5):
    colored = Image.fromarray(mask, mode="P")
    colored.putpalette(voc_palette())
    colored = colored.convert("RGB").resize(img_pil.size, Image.NEAREST)
    return Image.blend(img_pil.convert("RGB"), colored, alpha)

# 使用示例
img = Image.open("/root/vavae-upernet/data/VOCdevkit/VOC2012/JPEGImages/2007_000847.jpg")
pred_mask = predict_mask(model, img, keep_size=True)
save_color_mask(pred_mask, "demo_mask5.png")
overlay = overlay_mask(img, pred_mask, alpha=0.5)
overlay.save("demo_overlay5.png")
