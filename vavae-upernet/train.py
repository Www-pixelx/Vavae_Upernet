import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as VF
from transformers import UperNetForSemanticSegmentation
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from encoder import *  
import torch.optim.lr_scheduler as sched

class PascalVOCSegmentationTransform:
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __call__(self, image, target):
        image = self.image_transform(image)
        target = self.label_transform(target)
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target

def image_transform_fn(img, transform):
    return transform(img, img)[0] 

def target_transform_fn(target):
    return target

def collate_fn(batch, image_size=256):
    transform = PascalVOCSegmentationTransform(image_size)
    images, targets = [], []
    for img, target in batch:
        img, target = transform(img, target) 
        images.append(img)
        targets.append(target)
    return torch.stack(images), torch.stack(targets)

def get_voc_dataloaders(batch_size=8, image_size=256):
    # 训练集
    train_dataset = datasets.VOCSegmentation(
    root="/root/vavae-upernet/data",
    year="2012",
    image_set="train",
    download=False
    )

    val_dataset = datasets.VOCSegmentation(
    root="/root/vavae-upernet/data",
    year="2012",
    image_set="val",
    download=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    return train_loader, val_loader

# VAE Encoder + UperNet Decode Head
class FeatureAdapter(nn.Module):
    """适配VAE Encoder特征与UperNet解码器"""
    def __init__(self, encoder_feat_specs, upernet_feat_specs):
        super().__init__()
        self.adapters = nn.ModuleList()
        for (enc_c, enc_h, enc_w), (up_c, up_h, up_w) in zip(encoder_feat_specs, upernet_feat_specs):
            ops = []
            # 通道数调整
            ops.append(nn.Conv2d(enc_c, up_c, kernel_size=1, stride=1, padding=0))
            # 分辨率调整
            if (enc_h != up_h) or (enc_w != up_w):
                ops.append(nn.Upsample(size=(up_h, up_w), mode='bilinear', align_corners=True))
            self.adapters.append(nn.Sequential(*ops))

    def forward(self, feats):
        return [adapter(feat) for adapter, feat in zip(self.adapters, feats)]

def build_segmentation_model(num_classes=21, image_size=256):
    encoder = Encoder(
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        resolution=image_size,
        z_channels=32,
        num_res_blocks=2,
        attn_resolutions=(16,)
    )

    with torch.no_grad():
        test_input = torch.randn(1, 3, image_size, image_size)
        encoder_feats = encoder(test_input, return_multiscale_feats=True)
        encoder_specs = [(f.shape[1], f.shape[2], f.shape[3]) for f in encoder_feats]

    upernet = UperNetForSemanticSegmentation.from_pretrained(
        "/root/vavae-upernet/hf_models",
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # 允许类别头从 150 -> 21
    )

    if not hasattr(upernet, "decode_head"):
        raise RuntimeError("当前 transformers.UperNetForSemanticSegmentation 不暴露 decode_head，无法直连自定义 backbone。")

    upernet_specs = [
        (96,  image_size // 4,  image_size // 4),   # 1/4
        (192, image_size // 8,  image_size // 8),   # 1/8
        (384, image_size // 16, image_size // 16),  # 1/16
        (768, image_size // 32, image_size // 32),  # 1/32
    ]

    adapter = FeatureAdapter(encoder_specs, upernet_specs)

    class SegmentationModel(nn.Module):
        def __init__(self, encoder, adapter, upernet, out_size):
            super().__init__()
            self.encoder = encoder
            self.adapter = adapter
            self.decode_head = upernet.decode_head
            self.out_size = out_size  # (H, W)
           
        def forward(self, x):
            enc_feats = self.encoder(x, return_multiscale_feats=True)

            if len(enc_feats) >= 4:
                enc_feats = enc_feats[-4:]
            else:
                while len(enc_feats) < 4:
                    enc_feats = [enc_feats[0]] + enc_feats

            feats = self.adapter(enc_feats) 

            logits = self.decode_head(feats)  # [N, num_classes, H', W']

            if logits.shape[-2:] != self.out_size:
                logits = nn.functional.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=True)
            return logits

    return SegmentationModel(encoder, adapter, upernet, out_size=(image_size, image_size))

# 训练与验证工具
class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epochs, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        self.scheduler = sched.PolynomialLR(self.optimizer, total_iters=num_epochs, power=0.9)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)  # [N, C, H, W]

            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=True)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Validation]")
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=True)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        mIoU = self.calculate_mIoU(all_preds, all_targets, num_classes=21, ignore_index=255)
        return total_loss / len(self.val_loader), mIoU

    def calculate_mIoU(self, preds, targets, num_classes, ignore_index=255):
        ious = []
        for cls in range(num_classes):
            valid = (targets != ignore_index)
            pred_mask = (preds == cls) & valid
            target_mask = (targets == cls) & valid
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            if union == 0:
                ious.append(0.0)
            else:
                ious.append(intersection / union)
        return float(np.mean(ious))

    def train(self, num_epochs=30):
        best_mIoU = 0.0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mIoU = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f}")

            # 保存最佳模型
            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                torch.save(self.model.state_dict(), "best_segmentation_model.pth")
                print(f"Saved best model (mIoU: {best_mIoU:.4f})")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    image_size = 256
    num_epochs = 30

    print("Loading Pascal VOC dataset...")
    train_loader, val_loader = get_voc_dataloaders(batch_size, image_size)

    print("Building segmentation model...")
    model = build_segmentation_model(num_classes=21, image_size=image_size)  # Pascal VOC 有 21 类（含背景）

    print("Starting training...")

    trainer = Trainer(model, train_loader, val_loader, num_epochs, device)
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()


