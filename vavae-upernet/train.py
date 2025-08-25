import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as VF
from transformers import UperNetForSemanticSegmentation, AutoImageProcessor
from tqdm import tqdm
import numpy as np
from encoder import Encoder
import torch.optim.lr_scheduler as sched
from PIL import Image

# 配置参数集中管理
CONFIG = {
    "data_root": "/root/vavae-upernet/data",
    "upernet_path": "/root/vavae-upernet/hf_models",
    "processor_path": "/root/vavae-upernet/upernet-convnext-base",
    "batch_size": 4,
    "image_size": 256,
    "num_classes": 21,
    "num_epochs": 30,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

processor = AutoImageProcessor.from_pretrained(CONFIG["processor_path"])

class PascalVOCSegmentationTransform:
    def __init__(self, processor, image_size):
        self.processor = processor
        self.image_size = image_size

    def __call__(self, image, target):
      
        img_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    
        target_resized = target.resize((self.image_size, self.image_size), Image.NEAREST)
        target_tensor = torch.as_tensor(np.array(target_resized), dtype=torch.long)
        return img_tensor, target_tensor

def get_voc_dataloaders():
    transform = PascalVOCSegmentationTransform(processor, CONFIG["image_size"])
    
    def collate_fn(batch):
        images, targets = [], []
        for img, tgt in batch:
            img_t, tgt_t = transform(img, tgt)
            images.append(img_t)
            targets.append(tgt_t)
        return torch.stack(images), torch.stack(targets)
    
    def get_dataset(image_set):
        return datasets.VOCSegmentation(
            root=CONFIG["data_root"],
            year="2012",
            image_set=image_set,
            download=False
        )
    
    return (
        DataLoader(get_dataset("train"), CONFIG["batch_size"], True, num_workers=4, collate_fn=collate_fn),
        DataLoader(get_dataset("val"), CONFIG["batch_size"], False, num_workers=4, collate_fn=collate_fn)
    )

class FeatureAdapter(nn.Module):
    def __init__(self, encoder_specs, upernet_specs):
        super().__init__()
        self.adapters = nn.ModuleList()
        for (enc_c, enc_h, enc_w), (up_c, up_h, up_w) in zip(encoder_specs, upernet_specs):
            layers = [nn.Conv2d(enc_c, up_c, kernel_size=1)]  # 通道调整
            if (enc_h, enc_w) != (up_h, up_w):  # 分辨率调整
                layers.append(nn.Upsample(size=(up_h, up_w), mode='bilinear', align_corners=True))
            self.adapters.append(nn.Sequential(*layers))

    def forward(self, feats):
        return [adapter(feat) for adapter, feat in zip(self.adapters, feats)]


def build_segmentation_model():

    encoder = Encoder(
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        resolution=CONFIG["image_size"],
        z_channels=32,
        num_res_blocks=2,
        attn_resolutions=(16,)
    )
    

    with torch.no_grad():
        test_feat = encoder(torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"]), return_multiscale_feats=True)
        encoder_specs = [(f.shape[1], f.shape[2], f.shape[3]) for f in test_feat]
    

    upernet = UperNetForSemanticSegmentation.from_pretrained(
        CONFIG["upernet_path"],
        num_labels=CONFIG["num_classes"],
        ignore_mismatched_sizes=True
    )
    
    sizes = [CONFIG["image_size"] // s for s in [4, 8, 16, 32]]
    upernet_specs = [(96, sizes[0], sizes[0]), (192, sizes[1], sizes[1]),
                    (384, sizes[2], sizes[2]), (768, sizes[3], sizes[3])]
    
    adapter = FeatureAdapter(encoder_specs, upernet_specs)
    
    class SegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.adapter = adapter
            self.decode_head = upernet.decode_head
            self.out_size = (CONFIG["image_size"], CONFIG["image_size"])
        
        def forward(self, x):
            enc_feats = self.encoder(x, return_multiscale_feats=True)
            # 确保特征数量为4
            enc_feats = enc_feats[-4:] if len(enc_feats)>=4 else [enc_feats[0]]*(4-len(enc_feats)) + enc_feats
            feats = self.adapter(enc_feats)
            logits = self.decode_head(feats)
            # 确保输出尺寸正确
            return nn.functional.interpolate(logits, self.out_size, mode="bilinear", align_corners=True)
    
    return SegModel()

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(CONFIG["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG["lr"],
            momentum=CONFIG["momentum"],
            weight_decay=CONFIG["weight_decay"]
        )
        self.scheduler = sched.PolynomialLR(self.optimizer, total_iters=CONFIG["num_epochs"], power=0.9)

    def _process_batch(self, images, targets):
        """处理批次数据的公共方法"""
        images = images.to(CONFIG["device"])
        targets = targets.to(CONFIG["device"])
        outputs = self.model(images)
        return outputs, targets

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for images, targets in pbar:
            self.optimizer.zero_grad()
            outputs, targets = self._process_batch(images, targets)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/(pbar.n+1))
        
        return total_loss/len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Validation]")
            for images, targets in pbar:
                outputs, targets = self._process_batch(images, targets)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                pbar.set_postfix(loss=total_loss/(pbar.n+1))
        
        # 计算mIoU
        preds = np.concatenate(all_preds)
        tgts = np.concatenate(all_targets)
        mIoU = self._calculate_mIoU(preds, tgts)
        return total_loss/len(self.val_loader), mIoU

    def _calculate_mIoU(self, preds, targets):
        """向量化计算mIoU"""
        valid = (targets != 255)
        ious = []
        for cls in range(CONFIG["num_classes"]):
            pred_mask = (preds == cls) & valid
            tgt_mask = (targets == cls) & valid
            intersection = np.logical_and(pred_mask, tgt_mask).sum()
            union = np.logical_or(pred_mask, tgt_mask).sum()
            ious.append(intersection/union if union !=0 else 0.0)
        return np.mean(ious)

    def train(self):
        best_mIoU = 0.0
        for epoch in range(CONFIG["num_epochs"]):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mIoU = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f}")
            
            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                torch.save(self.model.state_dict(), "best_segmentation_model.pth")
                print(f"Saved best model (mIoU: {best_mIoU:.4f})")

# 主函数简化
def main():
    print("Loading dataset...")
    train_loader, val_loader = get_voc_dataloaders()
    
    print("Building model...")
    model = build_segmentation_model()
    
    print("Starting training...")
    Trainer(model, train_loader, val_loader).train()

if __name__ == "__main__":
    main()




