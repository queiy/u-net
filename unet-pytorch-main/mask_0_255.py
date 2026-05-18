import os
import numpy as np
from PIL import Image


def visualize_matched_labels(pred_dir, label_dir, output_dir):
    """
    根据预测图匹配标签，并将 [0, 1] 转换为 [0, 255] 保存
    """
    # 1. 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 2. 获取预测文件夹中的所有图像文件名（不含后缀）
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    pred_names = {os.path.splitext(f)[0]: f for f in pred_files}

    # 3. 扫描标签文件夹
    label_files_all = os.listdir(label_dir)
    # 建立 标签主文件名 -> 完整路径 的映射
    label_map = {os.path.splitext(f)[0]: f for f in label_files_all}

    print(f"开始处理，预测图共 {len(pred_names)} 张...")

    count = 0
    for main_name, p_file in pred_names.items():
        if main_name in label_map:
            # 匹配成功，获取标签完整文件名
            l_file = label_map[main_name]

            # 读取标签图像
            label_path = os.path.join(label_dir, l_file)
            # 使用 convert('L') 确保读取为 8位灰度模式
            label_img = Image.open(label_path).convert('L')
            label_array = np.array(label_img)

            # --- 关键转换逻辑 ---
            # 如果最大值就是 1，说明是 [0, 1] 索引图，需要乘以 255
            # 如果最大值已经是 255，则保持不变（防止重复操作）
            if label_array.max() <= 1:
                vis_array = (label_array * 255).astype(np.uint8)
            else:
                vis_array = label_array.astype(np.uint8)

            # 转换为图像并保存
            vis_img = Image.fromarray(vis_array)
            # 统一保存为 png，文件名与预测图保持一致
            save_path = os.path.join(output_dir, f"{main_name}.png")
            vis_img.save(save_path)

            count += 1
            if count % 10 == 0:
                print(f"已转换: {count} / {len(pred_names)}")
        else:
            print(f"警告: 未找到 {p_file} 对应的标签图像。")

    print(f"\n处理完成！共保存 {count} 张可视化标签到: {output_dir}")


# ---------------- 配置路径 ----------------
PRED_PATH = r'F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\output_masks_fp32'  # 你的 96 张预测图所在文件夹
LABEL_PATH = r'C:\dataset\VOCdevkit\VOC2007\SegmentationClass'  # 原始标签库（包含很多图）
SAVE_PATH = r'F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\output_masks_255'  # 新的存储文件夹（255倍后的图）

if __name__ == "__main__":
    visualize_matched_labels(PRED_PATH, LABEL_PATH, SAVE_PATH)