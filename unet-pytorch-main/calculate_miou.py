# import numpy as np
# import os
# from PIL import Image
#
#
# def fast_hist(a, b, n):
#     """计算混淆矩阵"""
#     k = (a >= 0) & (a < n)
#     return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
#
#
# def calculate_ore_iou_with_matching(pred_dir, label_dir, batch_size=16):
#     num_classes = 2  # 0:背景, 1:矿石前景
#
#     # 1. 获取预测文件夹中的所有文件名
#     pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#
#     # 2. 建立标签文件夹的文件映射（主文件名 -> 完整文件名）
#     # 这样可以快速根据预测图的文件名找到对应的标签图，即使后缀不同也可以匹配
#     label_files_all = os.listdir(label_dir)
#     label_map = {os.path.splitext(f)[0]: f for f in label_files_all}
#
#     # 3. 进行文件名匹配并配对
#     matched_pairs = []
#     for pf in pred_files:
#         main_name = os.path.splitext(pf)[0]  # 提取不带后缀的文件名
#         if main_name in label_map:
#             matched_pairs.append((pf, label_map[main_name]))
#
#     # 对配对列表进行排序，确保每次运行的分组是一致的（对学术论文的可重复性很重要）
#     matched_pairs.sort()
#
#     total_matched = len(matched_pairs)
#     if total_matched < 96:
#         print(f"警告：仅匹配到 {total_matched} 对图像，不足 96 张。将按实际数量计算。")
#     elif total_matched > 96:
#         print(f"匹配到 {total_matched} 对图像，将取前 96 张进行 6 组拆分计算。")
#         matched_pairs = matched_pairs[:96]
#
#     num_batches = len(matched_pairs) // batch_size
#
#     print(f"成功匹配到 {len(matched_pairs)} 张预测图及其对应标签。")
#     print(f"每组包含 {batch_size} 张图像，共计计算 {num_batches} 组。\n")
#
#     all_ore_ious = []
#
#     # 4. 分组计算 IoU
#     for i in range(num_batches):
#         batch_data = matched_pairs[i * batch_size: (i + 1) * batch_size]
#         batch_hist = np.zeros((num_classes, num_classes))
#
#         for p_file, l_file in batch_data:
#             # 读取图像并转为单通道灰度图
#             pred_img = Image.open(os.path.join(pred_dir, p_file)).convert('L')
#             label_img = Image.open(os.path.join(label_dir, l_file)).convert('L')
#
#             # 转为 numpy 数组
#             pred = np.array(pred_img)
#             label = np.array(label_img)
#
#             # 阈值处理：将图像转为二值索引 (0:背景, 1:前景)
#             # 假设你的标签/预测图中白色区域值较大（如255）
#             pred = (pred > 128).astype(np.uint8)
#             label = (label > 128).astype(np.uint8)
#
#             batch_hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
#
#         # 5. 计算 IoU
#         # 分母：TP + FP + FN
#         divisor = (batch_hist.sum(1) + batch_hist.sum(0) - np.diag(batch_hist))
#         # 避免分母为0
#         ious = np.diag(batch_hist) / (divisor + 1e-10)
#
#         ore_iou = ious[1]  # 获取索引为1的前景（矿石）IoU
#         all_ore_ious.append(ore_iou)
#
#         print(f"--- 第 {i + 1} 组 (Batch {i + 1}) ---")
#         print(f"包含图像: {batch_data[0][0]} ... {batch_data[-1][0]}")
#         print(f"本组矿石(Ore) IoU: {ore_iou:.4f}\n")
#
#     # 6. 输出总平均值
#     if all_ore_ious:
#         print("=" * 40)
#         print(f"96张图像 (6组) 平均矿石 IoU: {np.mean(all_ore_ious):.4f}")
#         print("=" * 40)
#
#
# # ---------------- 配置路径 ----------------
# PRED_PATH = r'F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\output_masks_fp16'  # 96张预测图所在路径
# LABEL_PATH = r'F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\output_masks_255'  # 包含大量标签图的路径
# BATCH_SIZE = 16  # 每组16张，96张刚好分6组
#
# if __name__ == "__main__":
#     calculate_ore_iou_with_matching(PRED_PATH, LABEL_PATH, BATCH_SIZE)


import numpy as np
import os
from PIL import Image


def fast_hist(a, b, n):
    """计算混淆矩阵"""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def calculate_ore_metrics_with_matching(pred_dir, label_dir, batch_size=16):
    num_classes = 2  # 0:背景, 1:矿石前景

    # 1. 获取预测文件夹中的所有文件名
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 2. 建立标签文件夹的文件映射
    label_files_all = os.listdir(label_dir)
    label_map = {os.path.splitext(f)[0]: f for f in label_files_all}

    # 3. 进行文件名匹配并配对
    matched_pairs = []
    for pf in pred_files:
        main_name = os.path.splitext(pf)[0]
        if main_name in label_map:
            matched_pairs.append((pf, label_map[main_name]))

    matched_pairs.sort()

    total_matched = len(matched_pairs)
    if total_matched < 96:
        print(f"警告：仅匹配到 {total_matched} 对图像，不足 96 张。按实际数量计算。")
    elif total_matched > 96:
        print(f"匹配到 {total_matched} 对图像，将取前 96 张进行分组计算。")
        matched_pairs = matched_pairs[:96]

    num_batches = len(matched_pairs) // batch_size

    print(f"成功匹配到 {len(matched_pairs)} 张预测图及其对应标签。")
    print(f"每组包含 {batch_size} 张图像，共计计算 {num_batches} 组。\n")

    # 用于存储每组的矿石类指标
    all_metrics = {
        "iou": [],
        "precision": [],
        "recall": [],
        "f_score": []
    }

    # 4. 分组计算
    for i in range(num_batches):
        batch_data = matched_pairs[i * batch_size: (i + 1) * batch_size]
        batch_hist = np.zeros((num_classes, num_classes))

        for p_file, l_file in batch_data:
            pred_img = Image.open(os.path.join(pred_dir, p_file)).convert('L')
            label_img = Image.open(os.path.join(label_dir, l_file)).convert('L')

            pred = np.array(pred_img)
            label = np.array(label_img)

            # 阈值处理 (0:背景, 1:前景)
            pred = (pred > 128).astype(np.uint8)
            label = (label > 128).astype(np.uint8)

            batch_hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # --- 核心指标计算 ---
        # 对于矿石类(索引为1):
        # TP = batch_hist[1, 1]
        # FP = batch_hist[0, 1]
        # FN = batch_hist[1, 0]
        # TN = batch_hist[0, 0]

        tp = batch_hist[1, 1]
        fp = batch_hist[0, 1]
        fn = batch_hist[1, 0]

        # 1. IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-10)

        # 2. Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-10)

        # 3. Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-10)

        # 4. F1-Score = 2 * P * R / (P + R)
        f_score = (2 * precision * recall) / (precision + recall + 1e-10)

        all_metrics["iou"].append(iou)
        all_metrics["precision"].append(precision)
        all_metrics["recall"].append(recall)
        all_metrics["f_score"].append(f_score)

        print(f"--- 第 {i + 1} 组 (Batch {i + 1}) ---")
        print(f"矿石 IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f_score:.4f}")

    # 6. 输出总平均值 (对应你论文表 5-1 的定量评价)
    if all_metrics["iou"]:
        print("\n" + "=" * 60)
        print(f"最终平均定量对比结果 ({len(matched_pairs)}张图像平均):")
        print(f"平均矿石 IoU:      {np.mean(all_metrics['iou']):.4f}")
        print(f"平均矿石 Precision: {np.mean(all_metrics['precision']):.4f}")
        print(f"平均矿石 Recall:    {np.mean(all_metrics['recall']):.4f}")
        print(f"平均矿石 F-Score:   {np.mean(all_metrics['f_score']):.4f}")
        print("=" * 60)


# ---------------- 配置路径 ----------------
PRED_PATH = r'F:\unet\denseaspp\unet-pytorch-main\unet-pytorch-main\img_out'
LABEL_PATH = r'F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\output_masks_255'
BATCH_SIZE = 16

if __name__ == "__main__":
    calculate_ore_metrics_with_matching(PRED_PATH, LABEL_PATH, BATCH_SIZE)