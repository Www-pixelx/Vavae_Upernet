import os
import torch
import requests
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_multiscale_feats=False):
        temb = None  
        hs = [self.conv_in(x)] 

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
    
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h) 

            if i_level != self.num_resolutions - 1:
                h_down = self.down[i_level].downsample(hs[-1])
                hs.append(h_down) 

        h_mid = hs[-1]
        h_mid = self.mid.block_1(h_mid, temb)
        h_mid = self.mid.attn_1(h_mid)
        h_mid = self.mid.block_2(h_mid, temb)
        hs.append(h_mid)  

        if return_multiscale_feats:
            selected_feats = [
                hs[2],   # 分辨率128×128，通道数128
                hs[4],   # 分辨率64×64，通道数128
                hs[6],   # 分辨率32×32，通道数256
                hs[11]   # 分辨率8×8，通道数512
            ]
            return selected_feats
        
        else:
            h = self.norm_out(hs[-1])
            h = nonlinearity(h)
            h = self.conv_out(h)
            return h

'''
Loading Pascal VOC dataset...
Building segmentation model...
Some weights of UperNetForSemanticSegmentation were not initialized from the model checkpoint at /root/vavae-upernet/hf_models and are newly initialized because the shapes did not match:
- auxiliary_head.classifier.bias: found shape torch.Size([150]) in the checkpoint and torch.Size([21]) in the model instantiated
- auxiliary_head.classifier.weight: found shape torch.Size([150, 256, 1, 1]) in the checkpoint and torch.Size([21, 256, 1, 1]) in the model instantiated
- decode_head.classifier.bias: found shape torch.Size([150]) in the checkpoint and torch.Size([21]) in the model instantiated
- decode_head.classifier.weight: found shape torch.Size([150, 512, 1, 1]) in the checkpoint and torch.Size([21, 512, 1, 1]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training...
Epoch 0 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.39it/s, loss=1.31]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.07it/s, loss=1.28]
Epoch 1/30
Train Loss: 1.3093 | Val Loss: 1.2803 | Val mIoU: 0.0421
Saved best model (mIoU: 0.0421)
Epoch 1 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.46it/s, loss=1.18]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.10it/s, loss=1.18]
Epoch 2/30
Train Loss: 1.1752 | Val Loss: 1.1806 | Val mIoU: 0.0368
Epoch 2 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.46it/s, loss=1.12]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.10it/s, loss=1.15]
Epoch 3/30
Train Loss: 1.1239 | Val Loss: 1.1508 | Val mIoU: 0.0486
Saved best model (mIoU: 0.0486)
Epoch 3 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=1.08]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.46it/s, loss=1.15]
Epoch 4/30
Train Loss: 1.0848 | Val Loss: 1.1529 | Val mIoU: 0.0530
Saved best model (mIoU: 0.0530)
Epoch 4 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.46it/s, loss=1.04]
[Validation]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.08it/s, loss=1.1]
Epoch 5/30
Train Loss: 1.0388 | Val Loss: 1.0968 | Val mIoU: 0.0701
Saved best model (mIoU: 0.0701)
Epoch 5 [Train]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=1.01]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.98it/s, loss=1.07]
Epoch 6/30
Train Loss: 1.0064 | Val Loss: 1.0697 | Val mIoU: 0.0670
Epoch 6 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.977]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.89it/s, loss=1.07]
Epoch 7/30
Train Loss: 0.9765 | Val Loss: 1.0726 | Val mIoU: 0.0768
Saved best model (mIoU: 0.0768)
Epoch 7 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.949]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.01it/s, loss=1.17]
Epoch 8/30
Train Loss: 0.9491 | Val Loss: 1.1677 | Val mIoU: 0.0735
Epoch 8 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.925]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.69it/s, loss=1.03]
Epoch 9/30
Train Loss: 0.9247 | Val Loss: 1.0286 | Val mIoU: 0.0814
Saved best model (mIoU: 0.0814)
Epoch 9 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.887]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.99it/s, loss=1.02]
Epoch 10/30
Train Loss: 0.8868 | Val Loss: 1.0233 | Val mIoU: 0.0873
Saved best model (mIoU: 0.0873)
Epoch 10 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.854]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.10it/s, loss=1.23]
Epoch 11/30
Train Loss: 0.8541 | Val Loss: 1.2264 | Val mIoU: 0.0739
Epoch 11 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.827]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.99it/s, loss=1.08]
Epoch 12/30
Train Loss: 0.8273 | Val Loss: 1.0757 | Val mIoU: 0.0891
Saved best model (mIoU: 0.0891)
Epoch 12 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.777]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.92it/s, loss=1.09]
Epoch 13/30
Train Loss: 0.7766 | Val Loss: 1.0926 | Val mIoU: 0.0883
Epoch 13 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.75]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.70it/s, loss=1.12]
Epoch 14/30
Train Loss: 0.7503 | Val Loss: 1.1183 | Val mIoU: 0.0810
Epoch 14 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.704]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.88it/s, loss=1.16]
Epoch 15/30
Train Loss: 0.7036 | Val Loss: 1.1592 | Val mIoU: 0.1103
Saved best model (mIoU: 0.1103)
Epoch 15 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.663]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.77it/s, loss=1.11]
Epoch 16/30
Train Loss: 0.6626 | Val Loss: 1.1071 | Val mIoU: 0.1026
Epoch 16 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.589]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.88it/s, loss=1.12]
Epoch 17/30
Train Loss: 0.5892 | Val Loss: 1.1227 | Val mIoU: 0.1057
Epoch 17 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.555]
[Validation]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.96it/s, loss=1.1]
Epoch 18/30
Train Loss: 0.5547 | Val Loss: 1.1041 | Val mIoU: 0.1071
Epoch 18 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.497]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:12<00:00, 14.26it/s, loss=1.04]
Epoch 19/30
Train Loss: 0.4973 | Val Loss: 1.0437 | Val mIoU: 0.1327
Saved best model (mIoU: 0.1327)
Epoch 19 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.42it/s, loss=0.439]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.79it/s, loss=1.08]
Epoch 20/30
Train Loss: 0.4385 | Val Loss: 1.0837 | Val mIoU: 0.1217
Epoch 20 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.384]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.82it/s, loss=1.14]
Epoch 21/30
Train Loss: 0.3838 | Val Loss: 1.1437 | Val mIoU: 0.1069
Epoch 21 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.36]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.90it/s, loss=1.09]
Epoch 22/30
Train Loss: 0.3601 | Val Loss: 1.0921 | Val mIoU: 0.1295
Epoch 22 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.312]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.84it/s, loss=1.12]
Epoch 23/30
Train Loss: 0.3115 | Val Loss: 1.1153 | Val mIoU: 0.1332
Saved best model (mIoU: 0.1332)
Epoch 23 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.43it/s, loss=0.277]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.75it/s, loss=1.08]
Epoch 24/30
Train Loss: 0.2766 | Val Loss: 1.0821 | Val mIoU: 0.1387
Saved best model (mIoU: 0.1387)
Epoch 24 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.262]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.90it/s, loss=1.11]
Epoch 25/30
Train Loss: 0.2623 | Val Loss: 1.1115 | Val mIoU: 0.1330
Epoch 25 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.44it/s, loss=0.245]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.87it/s, loss=1.11]
Epoch 26/30
Train Loss: 0.2446 | Val Loss: 1.1105 | Val mIoU: 0.1371
Epoch 26 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.43it/s, loss=0.225]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.89it/s, loss=1.12]
Epoch 27/30
Train Loss: 0.2252 | Val Loss: 1.1217 | Val mIoU: 0.1348
Epoch 27 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.212]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.81it/s, loss=1.14]
Epoch 28/30
Train Loss: 0.2115 | Val Loss: 1.1380 | Val mIoU: 0.1354
Epoch 28 [Train]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.43it/s, loss=0.196]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.91it/s, loss=1.11]
Epoch 29/30
Train Loss: 0.1964 | Val Loss: 1.1134 | Val mIoU: 0.1385
Epoch 29 [Train]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 183/183 [00:33<00:00,  5.45it/s, loss=0.19]
[Validation]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 182/182 [00:13<00:00, 13.86it/s, loss=1.14]
Epoch 30/30
Train Loss: 0.1902 | Val Loss: 1.1391 | Val mIoU: 0.1382
'''