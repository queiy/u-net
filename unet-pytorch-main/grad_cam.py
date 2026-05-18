import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 模拟导入你的模型
from nets.unet import Unet  # 这里替换成你的 U-Net 定义
model = Unet(n_channels=3, n_classes=2)  # 输入通道数和类别数根据你的任务调整
model.eval()

# 加载权重
model.load_state_dict(torch.load("unet_weights.pth", map_location="cpu"))

# 输入图片
img = cv2.imread("test.jpg")  # H×W×3
img = cv2.resize(img, (256, 256))
input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

input_tensor.requires_grad = True

# ====== hook 记录特征图 & 梯度 ======
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# 挑一个卷积层，一般是编码器最后一层
target_layer = model.down4  # 根据你代码里U-Net的结构调整
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ====== 前向传播 ======
output = model(input_tensor)  # shape: [1, C, H, W]

# 选定目标
target_class = 1  # 目标类别
target_mask = output[0, target_class, :, :]  # shape: [H, W]

# 这里也可以选择某个像素点：
# target_mask = output[0, target_class, i, j]

# 目标的和（或均值）作为标量
score = target_mask.mean()

# ====== 反向传播 ======
model.zero_grad()
score.backward()

# ====== 计算 Grad-CAM ======
# 特征图和梯度
feature_map = feature_maps[0].detach().squeeze(0)  # [C, h, w]
grad = gradients[0].detach().squeeze(0)            # [C, h, w]

weights = torch.mean(grad, dim=(1, 2))  # [C]

# 加权求和
cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32)

for i, w in enumerate(weights):
    cam += w * feature_map[i]

# ReLU
cam = F.relu(cam)

# 归一化
cam -= cam.min()
cam /= cam.max()

# 上采样到输入大小
cam_np = cam.numpy()
cam_np = cv2.resize(cam_np, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255

overlay = heatmap + np.float32(img) / 255
overlay /= overlay.max()

# ====== 可视化 ======
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img[..., ::-1])
plt.title("Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam_np, cmap='jet')
plt.title("Grad-CAM")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay[..., ::-1])
plt.title("Overlay")
plt.axis("off")

plt.show()
