import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from train import build_segmentation_model, PascalVOCSegmentationTransform

PALETTE = [
    [0, 0, 0],  # 背景（class 0）
    [128, 0, 0],  # class 1
    [0, 128, 0],  # class 2
    [128, 128, 0],  # class 3
    [0, 0, 128],  # class 4
    [128, 0, 128],  # class 5
    [0, 128, 128],  # class 6
    [128, 128, 128],  # class 7
    [64, 0, 0],  # class 8
    [192, 0, 0],  # class 9
    [64, 128, 0],  # class 10
    [192, 128, 0],  # class 11
    [64, 0, 128],  # class 12
    [192, 0, 128],  # class 13
    [64, 128, 128],  # class 14
    [192, 128, 128],  # class 15
    [0, 64, 0],  # class 16
    [128, 64, 0],  # class 17
    [0, 192, 0],  # class 18
    [128, 192, 0],  # class 19
    [0, 64, 128]  # class 20
]


def apply_color_map(predictions):
    color_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    
    for i in range(21):  # 21 类
        color_image[predictions == i] = PALETTE[i]
    
    return color_image

model = build_segmentation_model(num_classes=21, image_size=256)  

model.load_state_dict(torch.load(r"/root/vavae-upernet/best_segmentation_model.pth"))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image_path = r"/root/vavae-upernet/data/VOCdevkit/VOC2012/JPEGImages/2007_000042.jpg"

image = Image.open(image_path)

transform = PascalVOCSegmentationTransform(image_size=256)  # 使用与训练时相同的大小

image_tensor, _ = transform(image, image)  # 这里只需要图像部分，目标标签用不到

image_tensor = image_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)  # 输出尺寸：[1, num_classes, H, W]

# 选择每个像素点的最大概率类别
predictions = torch.argmax(output, dim=1)  # [1, H, W]

# 转换为 NumPy 数组，去掉 batch 维度
predictions = predictions.squeeze().cpu().numpy()

# 将预测的类别索引图像转为彩色图像
color_predictions = apply_color_map(predictions)

# 将彩色图像转换为 PIL 图像并保存
output_image = Image.fromarray(color_predictions)
output_image.save("segmentation_result_color.png")

# 可视化原图和彩色预测结果
original_image = Image.open(image_path)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[1].imshow(color_predictions)
ax[1].set_title("Predicted Segmentation (Color)")
plt.show()
