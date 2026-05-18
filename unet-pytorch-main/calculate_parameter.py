# calc_params.py
import torch
import numpy as np
from nets.unet import Unet
from nets.unet_training import weights_init


def count_parameters(model):
    """统计总参数量与可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # ---------------------------------------------
    # 你可以根据需要手动修改以下配置
    # ---------------------------------------------
    num_classes = 2
    backbone = "resnet50"
    pretrained = False  # 是否加载默认预训练权重
    model_path = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\logs\loss_2025_07_25_17_34_29\best_epoch_weights.pth"  # 如果你需要加载训练好的权重，填路径，不需要就留空

    # ---------------------------------------------
    #  实例化模型
    # ---------------------------------------------
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)

    # 如果不加载预训练权重，则初始化
    if not pretrained:
        weights_init(model)

    # 加载自定义权重
    if model_path != "":
        print(f"Loading weights from: {model_path}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location="cpu")
        temp_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("权重加载完成！")

    # ---------------------------------------------
    # 统计参数量
    # ---------------------------------------------
    total_params, trainable_params = count_parameters(model)

    print("\n==============================")
    print("🔍 模型参数量统计结果")
    print("==============================")
    print(f"➡️  总参数量 Total Params       : {total_params:,}  ({total_params / 1e6:.2f} M)")
    print(f"➡️  可训练参数 Trainable Params : {trainable_params:,}  ({trainable_params / 1e6:.2f} M)")
    print("==============================\n")