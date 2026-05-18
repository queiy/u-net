import os
import numpy as np
from PIL import Image


def calculate_binary_mask_distribution(mask_dir):
    """修正溢出问题，统计二值掩码的像素分布"""
    total_pixels = np.int64(0)  # 用int64避免溢出
    background_pixels = np.int64(0)  # 背景像素（0）
    target_pixels = np.int64(0)  # 目标像素（1）

    for mask_name in os.listdir(mask_dir):
        if mask_name.endswith(('.png', '.jpg')):
            mask_path = os.path.join(mask_dir, mask_name)
            mask = np.array(Image.open(mask_path), dtype=np.uint8)  # 确保是uint8类型

            # 统计0和1的像素数（转换为int64避免中间结果溢出）
            bg_count = np.int64(np.sum(mask == 0))
            target_count = np.int64(np.sum(mask == 1))

            # 累加（用int64类型安全累加）
            background_pixels += bg_count
            target_pixels += target_count
            total_pixels += np.int64(mask.size)

    # 计算占比（处理total_pixels为0的极端情况）
    if total_pixels == 0:
        return 0.0, 0.0
    background_ratio = background_pixels / total_pixels
    target_ratio = target_pixels / total_pixels

    print(f"背景像素占比: {background_ratio:.2%}")
    print(f"目标像素占比: {target_ratio:.2%}")
    print(f"总像素校验: {background_ratio + target_ratio:.2%}（应为100%，否则存在其他像素值）")
    return background_ratio, target_ratio


# 使用示例
if __name__ == "__main__":
    mask_dir = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\VOCdevkit\VOC2007\SegmentationClass"  # 你的掩码文件夹路径
    bg_ratio, target_ratio = calculate_binary_mask_distribution(mask_dir)