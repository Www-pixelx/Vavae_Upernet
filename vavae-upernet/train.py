import os
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchmetrics
import segmentation_models_pytorch as smp
from transformers import UperNetForSemanticSegmentation, AutoImageProcessor
from torch.optim.lr_scheduler import LambdaLR

from encoder import Encoder


# ---------------- 配置 ----------------
CONFIG = {
    "data_root": "/root/vavae-upernet/data",
    "upernet_path": "/root/vavae-upernet/hf_models",
    "processor_path": "/root/vavae-upernet/upernet-convnext-base",

    "batch_size": 4,
    "image_size": 256,
    "num_classes": 21,

    "num_epochs": 50,
    "num_warmup": 5,
    "freeze_encoder_epochs": 5,      # 前若干 epoch 冻结 encoder
    "early_stop_patience": 12,       # 连续 N 个 epoch mIoU 未提升则早停
    "grad_accum_steps": 1,           # 梯度累积步数
    "label_smoothing": 0.05,         # 交叉熵 label smoothing（0 关闭）

    "lr": 2e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 10.0,

    "seed": 42,
    "deterministic": False,

    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ---------------- 随机种子与后端 ----------------
def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ---------------- 预处理器均值方差 ----------------
processor = AutoImageProcessor.from_pretrained(CONFIG["processor_path"])
MEAN = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
STD = getattr(processor, "image_std", [0.229, 0.224, 0.225])


# ---------------- 数据增强 ----------------
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"] + 32, CONFIG["image_size"] + 32),
    A.RandomCrop(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.3, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])


# ---------------- 数据集封装 ----------------
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_set="train"):
        self.dataset = datasets.VOCSegmentation(
            root=CONFIG["data_root"], year="2012", image_set=image_set, download=False
        )
        self.transform = train_transform if image_set == "train" else val_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, tgt = self.dataset[idx]  # PIL.Image
        img_np = np.array(img)
        tgt_np = np.array(tgt)  # uint8, 0..20, 255为ignore
        augmented = self.transform(image=img_np, mask=tgt_np)
        image = augmented['image']
        mask = torch.as_tensor(augmented['mask'], dtype=torch.long)
        return image, mask


def get_voc_dataloaders():
    pin = torch.cuda.is_available()
    num_workers = min(8, os.cpu_count() or 4)

    train_ds = VOCDataset("train")
    val_ds = VOCDataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
        drop_last=False
    )
    return train_loader, val_loader


# ---------------- FeatureAdapter ----------------

class FeatureAdapter(nn.Module):
    def __init__(self, encoder_specs, upernet_specs):
        super().__init__()
        self.adapters = nn.ModuleList()
        for (enc_c, enc_h, enc_w), (up_c, up_h, up_w) in zip(encoder_specs, upernet_specs):
            upsample_size = (up_h, up_w) if (enc_h, enc_w) != (up_h, up_w) else None
            self.adapters.append(ResidualAdapter(enc_c, up_c, upsample_size))

    def forward(self, feats):
        return [adapter(feat) for adapter, feat in zip(self.adapters, feats)]


class ResidualAdapter(nn.Module):
    def __init__(self, in_ch, out_ch, up_size=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.up_size = up_size
        # 如果输入通道和输出通道不同，用 1x1 Conv 调整残差
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.res_conv(x)
        if self.up_size:
            x = F.interpolate(x, size=self.up_size, mode='bilinear', align_corners=False)
            identity = F.interpolate(identity, size=self.up_size, mode='bilinear', align_corners=False)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + identity

# ---------------- 构建模型 ----------------
def build_segmentation_model():
    # Encoder
    encoder = Encoder(
        ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2, attn_resolutions=(16,),
        in_channels=3, resolution=CONFIG["image_size"], z_channels=32
    )
    ckpt_path = '/root/vavae-upernet/vavae-imagenet256-f16d32-dinov2.pt'
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state_dict, strict=False)

    # 探测 encoder 输出规格（从大到小排序并取4个）
    encoder.eval()
    with torch.no_grad():
        test_feat = encoder(
            torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"]),
            return_multiscale_feats=True
        )
        test_feat = sorted(test_feat, key=lambda f: f.shape[-1], reverse=True)
        test_feat = test_feat[:4] if len(test_feat) >= 4 else ([test_feat[0]] * (4 - len(test_feat)) + test_feat)
        encoder_specs = [(f.shape[1], f.shape[2], f.shape[3]) for f in test_feat]

    # UPerNet 解码头
    upernet = UperNetForSemanticSegmentation.from_pretrained(
        CONFIG["upernet_path"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True
    )

    # 期望的金字塔特征通道与分辨率（ConvNeXt-B：96/192/384/768）
    sizes = [CONFIG["image_size"] // s for s in [4, 8, 16, 32]]
    upernet_specs = [
        (96, sizes[0], sizes[0]),
        (192, sizes[1], sizes[1]),
        (384, sizes[2], sizes[2]),
        (768, sizes[3], sizes[3])
    ]

    adapter = FeatureAdapter(encoder_specs, upernet_specs)

    class SegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.adapter = adapter
            self.decode_head = upernet.decode_head  # 直接使用 UPerNet 的 decode head
            self.out_size = (CONFIG["image_size"], CONFIG["image_size"])

        def forward(self, x):
            enc_feats = self.encoder(x, return_multiscale_feats=True)
            enc_feats = sorted(enc_feats, key=lambda f: f.shape[-1], reverse=True)
            enc_feats = enc_feats[:4] if len(enc_feats) >= 4 else ([enc_feats[0]] * (4 - len(enc_feats)) + enc_feats)
            feats = self.adapter(enc_feats)  # list[tensor(B,C,H,W)] x4
            logits = self.decode_head(feats)  # (B, num_classes, h, w)
            logits = nn.functional.interpolate(logits, self.out_size, mode="bilinear", align_corners=False)
            return logits

    return SegModel()


# ---------------- 训练器 ----------------
class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.device = CONFIG["device"]
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 损失函数：CE + Dice
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=255,
            label_smoothing=CONFIG["label_smoothing"]
        )
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True,
            ignore_index=255
        )

        # AMP
        self.is_cuda = self.device.startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.is_cuda)

        # 优化器：分组学习率
        encoder_params = list(model.encoder.parameters())
        adapter_params = list(model.adapter.parameters())
        head_params = list(model.decode_head.parameters())
        self.optimizer = optim.AdamW([
            {"params": encoder_params, "lr": CONFIG["lr"] * 0.1},
            {"params": adapter_params, "lr": CONFIG["lr"]},
            {"params": head_params, "lr": CONFIG["lr"] * 10}
        ], weight_decay=CONFIG["weight_decay"])

        # ---------------- Cosine Annealing + Warmup ----------------
        self.warmup_epochs = CONFIG["num_warmup"]
        self.cosine_epochs = CONFIG["num_epochs"] - self.warmup_epochs
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # 线性 warmup
                return [(epoch + 1) / self.warmup_epochs for _ in self.base_lrs]
            else:
                # Cosine Annealing
                t = (epoch - self.warmup_epochs) / max(1, self.cosine_epochs)
                return [0.5 * (1 + math.cos(math.pi * t)) for _ in self.base_lrs]

        # LambdaLR 支持返回 list 对应 param_groups
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # 指标
        self.metric = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=CONFIG["num_classes"],
            ignore_index=255
        ).to(self.device)

        # 训练控制
        self.best_mIoU = 0.0
        self.no_improve_epochs = 0

    def set_encoder_freeze(self, freeze=True):
        for p in self.model.encoder.parameters():
            p.requires_grad = not freeze

    def _forward_loss(self, images, targets):
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.is_cuda):
            outputs = self.model(images)
            ce = self.ce_loss(outputs, targets)
            dice = self.dice_loss(outputs, targets)
            loss = ce + 0.5 * dice
        return outputs, loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]")

        self.optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(pbar, 1):
            outputs, loss = self._forward_loss(images, targets)
            loss_to_scale = loss / CONFIG.get("grad_accum_steps", 1)

            self.scaler.scale(loss_to_scale).backward()

            if step % CONFIG.get("grad_accum_steps", 1) == 0:
                # 梯度裁剪
                if CONFIG.get("max_grad_norm", None):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        CONFIG["max_grad_norm"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/step)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, tta_flip=True):
        self.model.eval()
        total_loss = 0.0
        self.metric.reset()

        pbar = tqdm(self.val_loader, desc="[Validation]")
        for step, (images, targets) in enumerate(pbar, 1):
            outputs, loss = self._forward_loss(images, targets)

            if tta_flip:
                flipped = torch.flip(images.to(self.device), dims=[-1])
                outputs_flip, _ = self._forward_loss(flipped, targets.to(self.device))
                outputs_flip = torch.flip(outputs_flip, dims=[-1])
                outputs = (outputs + outputs_flip) / 2.0

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.metric.update(preds, targets.to(self.device))
            pbar.set_postfix(loss=total_loss/step)

        return total_loss/len(self.val_loader), self.metric.compute().item()

    def train(self):
        # 初始冻结 encoder
        if CONFIG.get("freeze_encoder_epochs", 0) > 0:
            self.set_encoder_freeze(True)

        for epoch in range(CONFIG["num_epochs"]):
            if epoch == CONFIG.get("freeze_encoder_epochs", 0):
                self.set_encoder_freeze(False)

            train_loss = self.train_epoch(epoch)
            val_loss, val_mIoU = self.validate(tta_flip=True)

            self.scheduler.step()

            print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {100*val_mIoU:.2f}%")

            if val_mIoU > self.best_mIoU:
                self.best_mIoU = val_mIoU
                self.no_improve_epochs = 0
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                    "best_mIoU": self.best_mIoU
                }, "best_segmentation_model_v4.pth")
                print(f"Saved best model (mIoU: {100*self.best_mIoU:.2f}%)")
            else:
                self.no_improve_epochs += 1

            if self.no_improve_epochs >= CONFIG.get("early_stop_patience", 12):
                print(f"Early stopping at epoch {epoch+1}")
                break




# ---------------- 主函数 ----------------
def main():
    set_seed(CONFIG["seed"], CONFIG["deterministic"])
    print("Loading dataset...")
    train_loader, val_loader = get_voc_dataloaders()

    print("Building model...")
    model = build_segmentation_model()

    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()

'''
Epoch 1/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:16<00:00, 22.81it/s, loss=1.57]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:18<00:00, 19.89it/s, loss=1.28]
Epoch 1/50 | Train Loss: 1.5657 | Val Loss: 1.2783 | Val mIoU: 8.68%
Saved best model (mIoU: 8.68%)
Epoch 2/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:14<00:00, 25.30it/s, loss=1.44]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.47it/s, loss=1.22]
Epoch 2/50 | Train Loss: 1.4442 | Val Loss: 1.2228 | Val mIoU: 10.53%
Saved best model (mIoU: 10.53%)
Epoch 3/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:14<00:00, 25.35it/s, loss=1.39]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=1.21]
Epoch 3/50 | Train Loss: 1.3918 | Val Loss: 1.2105 | Val mIoU: 14.64%
Saved best model (mIoU: 14.64%)
Epoch 4/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:14<00:00, 25.31it/s, loss=1.35]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.43it/s, loss=1.17]
Epoch 4/50 | Train Loss: 1.3490 | Val Loss: 1.1665 | Val mIoU: 16.97%
Saved best model (mIoU: 16.97%)
Epoch 5/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:14<00:00, 25.31it/s, loss=1.29]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=1.14]
Epoch 5/50 | Train Loss: 1.2880 | Val Loss: 1.1363 | Val mIoU: 15.66%
Epoch 6/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.15it/s, loss=1.18]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.49it/s, loss=1.12]
Epoch 6/50 | Train Loss: 1.1764 | Val Loss: 1.1194 | Val mIoU: 20.56%
Saved best model (mIoU: 20.56%)
Epoch 7/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.41it/s, loss=1.12]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.50it/s, loss=1.02]
Epoch 7/50 | Train Loss: 1.1154 | Val Loss: 1.0168 | Val mIoU: 26.32%
Saved best model (mIoU: 26.32%)
Epoch 8/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.42it/s, loss=1.07]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.968]
Epoch 8/50 | Train Loss: 1.0657 | Val Loss: 0.9677 | Val mIoU: 29.13%
Saved best model (mIoU: 29.13%)
Epoch 9/50 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.41it/s, loss=1.02]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.959]
Epoch 9/50 | Train Loss: 1.0197 | Val Loss: 0.9591 | Val mIoU: 30.44%
Saved best model (mIoU: 30.44%)
Epoch 10/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.962]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.945]
Epoch 10/50 | Train Loss: 0.9616 | Val Loss: 0.9453 | Val mIoU: 33.31%
Saved best model (mIoU: 33.31%)
Epoch 11/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.35it/s, loss=0.932]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.41it/s, loss=0.924]
Epoch 11/50 | Train Loss: 0.9316 | Val Loss: 0.9241 | Val mIoU: 33.92%
Saved best model (mIoU: 33.92%)
Epoch 12/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.35it/s, loss=0.867]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.929]
Epoch 12/50 | Train Loss: 0.8672 | Val Loss: 0.9292 | Val mIoU: 34.39%
Saved best model (mIoU: 34.39%)
Epoch 13/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.849]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.44it/s, loss=0.889]
Epoch 13/50 | Train Loss: 0.8493 | Val Loss: 0.8885 | Val mIoU: 38.17%
Saved best model (mIoU: 38.17%)
Epoch 14/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.41it/s, loss=0.821]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.48it/s, loss=0.897]
Epoch 14/50 | Train Loss: 0.8208 | Val Loss: 0.8975 | Val mIoU: 38.87%
Saved best model (mIoU: 38.87%)
Epoch 15/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.794]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.874]
Epoch 15/50 | Train Loss: 0.7945 | Val Loss: 0.8736 | Val mIoU: 40.73%
Saved best model (mIoU: 40.73%)
Epoch 16/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.41it/s, loss=0.757]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.851]
Epoch 16/50 | Train Loss: 0.7571 | Val Loss: 0.8511 | Val mIoU: 41.76%
Saved best model (mIoU: 41.76%)
Epoch 17/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.37it/s, loss=0.735]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.859]
Epoch 17/50 | Train Loss: 0.7348 | Val Loss: 0.8591 | Val mIoU: 42.64%
Saved best model (mIoU: 42.64%)
Epoch 18/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.723]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.874]
Epoch 18/50 | Train Loss: 0.7228 | Val Loss: 0.8745 | Val mIoU: 42.31%
Epoch 19/50 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.38it/s, loss=0.69]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.44it/s, loss=0.86]
Epoch 19/50 | Train Loss: 0.6898 | Val Loss: 0.8600 | Val mIoU: 44.52%
Saved best model (mIoU: 44.52%)
Epoch 20/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.42it/s, loss=0.664]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.47it/s, loss=0.878]
Epoch 20/50 | Train Loss: 0.6642 | Val Loss: 0.8781 | Val mIoU: 41.06%
Epoch 21/50 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.66]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.47it/s, loss=0.837]
Epoch 21/50 | Train Loss: 0.6602 | Val Loss: 0.8374 | Val mIoU: 44.94%
Saved best model (mIoU: 44.94%)
Epoch 22/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.632]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.49it/s, loss=0.867]
Epoch 22/50 | Train Loss: 0.6318 | Val Loss: 0.8665 | Val mIoU: 44.02%
Epoch 23/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.38it/s, loss=0.609]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.43it/s, loss=0.845]
Epoch 23/50 | Train Loss: 0.6094 | Val Loss: 0.8450 | Val mIoU: 45.88%
Saved best model (mIoU: 45.88%)
Epoch 24/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.595]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.878]
Epoch 24/50 | Train Loss: 0.5951 | Val Loss: 0.8782 | Val mIoU: 45.40%
Epoch 25/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.35it/s, loss=0.576]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.849]
Epoch 25/50 | Train Loss: 0.5761 | Val Loss: 0.8486 | Val mIoU: 45.48%
Epoch 26/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.37it/s, loss=0.562]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.851]
Epoch 26/50 | Train Loss: 0.5615 | Val Loss: 0.8511 | Val mIoU: 46.20%
Saved best model (mIoU: 46.20%)
Epoch 27/50 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.40it/s, loss=0.55]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.49it/s, loss=0.832]
Epoch 27/50 | Train Loss: 0.5500 | Val Loss: 0.8324 | Val mIoU: 46.62%
Saved best model (mIoU: 46.62%)
Epoch 28/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.37it/s, loss=0.535]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.50it/s, loss=0.849]
Epoch 28/50 | Train Loss: 0.5353 | Val Loss: 0.8488 | Val mIoU: 47.19%
Saved best model (mIoU: 47.19%)
Epoch 29/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.40it/s, loss=0.522]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:31<00:00, 11.53it/s, loss=0.844]
Epoch 29/50 | Train Loss: 0.5225 | Val Loss: 0.8445 | Val mIoU: 48.03%
Saved best model (mIoU: 48.03%)
Epoch 30/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.42it/s, loss=0.522]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.51it/s, loss=0.828]
Epoch 30/50 | Train Loss: 0.5217 | Val Loss: 0.8285 | Val mIoU: 47.73%
Epoch 31/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.37it/s, loss=0.504]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.824]
Epoch 31/50 | Train Loss: 0.5037 | Val Loss: 0.8236 | Val mIoU: 48.99%
Saved best model (mIoU: 48.99%)
Epoch 32/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.40it/s, loss=0.496]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.47it/s, loss=0.82]
Epoch 32/50 | Train Loss: 0.4957 | Val Loss: 0.8199 | Val mIoU: 48.24%
Epoch 33/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.485]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.824]
Epoch 33/50 | Train Loss: 0.4846 | Val Loss: 0.8236 | Val mIoU: 48.59%
Epoch 34/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.477]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.44it/s, loss=0.842]
Epoch 34/50 | Train Loss: 0.4767 | Val Loss: 0.8418 | Val mIoU: 47.91%
Epoch 35/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.29it/s, loss=0.474]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.42it/s, loss=0.83]
Epoch 35/50 | Train Loss: 0.4740 | Val Loss: 0.8299 | Val mIoU: 47.72%
Epoch 36/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.34it/s, loss=0.465]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.35it/s, loss=0.82]
Epoch 36/50 | Train Loss: 0.4647 | Val Loss: 0.8199 | Val mIoU: 48.34%
Epoch 37/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.459]
[Validation]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.44it/s, loss=0.82]
Epoch 37/50 | Train Loss: 0.4593 | Val Loss: 0.8205 | Val mIoU: 49.41%
Saved best model (mIoU: 49.41%)
Epoch 38/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.28it/s, loss=0.449]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.39it/s, loss=0.807]
Epoch 38/50 | Train Loss: 0.4495 | Val Loss: 0.8074 | Val mIoU: 49.96%
Saved best model (mIoU: 49.96%)
Epoch 39/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.32it/s, loss=0.446]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.35it/s, loss=0.812]
Epoch 39/50 | Train Loss: 0.4464 | Val Loss: 0.8118 | Val mIoU: 49.85%
Epoch 40/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.30it/s, loss=0.438]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.40it/s, loss=0.818]
Epoch 40/50 | Train Loss: 0.4381 | Val Loss: 0.8184 | Val mIoU: 49.35%
Epoch 41/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.31it/s, loss=0.439]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.42it/s, loss=0.813]
Epoch 41/50 | Train Loss: 0.4394 | Val Loss: 0.8132 | Val mIoU: 49.23%
Epoch 42/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.432]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.47it/s, loss=0.808]
Epoch 42/50 | Train Loss: 0.4324 | Val Loss: 0.8076 | Val mIoU: 50.15%
Saved best model (mIoU: 50.15%)
Epoch 43/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.431]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.40it/s, loss=0.798]
Epoch 43/50 | Train Loss: 0.4308 | Val Loss: 0.7977 | Val mIoU: 50.82%
Saved best model (mIoU: 50.82%)
Epoch 44/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.37it/s, loss=0.423]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.45it/s, loss=0.799]
Epoch 44/50 | Train Loss: 0.4233 | Val Loss: 0.7989 | Val mIoU: 50.74%
Epoch 45/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.39it/s, loss=0.422]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.46it/s, loss=0.798]
Epoch 45/50 | Train Loss: 0.4220 | Val Loss: 0.7977 | Val mIoU: 50.95%
Saved best model (mIoU: 50.95%)
Epoch 46/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.38it/s, loss=0.419]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.40it/s, loss=0.799]
Epoch 46/50 | Train Loss: 0.4186 | Val Loss: 0.7995 | Val mIoU: 50.95%
Saved best model (mIoU: 50.95%)
Epoch 47/50 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.36it/s, loss=0.42]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:58<00:00,  6.20it/s, loss=0.789]
Epoch 47/50 | Train Loss: 0.4200 | Val Loss: 0.7891 | Val mIoU: 50.90%
Epoch 48/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [01:03<00:00,  5.77it/s, loss=0.416]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:44<00:00,  8.07it/s, loss=0.793]
Epoch 48/50 | Train Loss: 0.4164 | Val Loss: 0.7934 | Val mIoU: 51.05%
Saved best model (mIoU: 51.05%)
Epoch 49/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.42it/s, loss=0.412]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.54it/s, loss=0.788]
Epoch 49/50 | Train Loss: 0.4123 | Val Loss: 0.7884 | Val mIoU: 51.31%
Saved best model (mIoU: 51.31%)
Epoch 50/50 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 366/366 [00:25<00:00, 14.40it/s, loss=0.409]
[Validation]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 363/363 [00:17<00:00, 20.52it/s, loss=0.791]
Epoch 50/50 | Train Loss: 0.4092 | Val Loss: 0.7914 | Val mIoU: 50.50%
'''
