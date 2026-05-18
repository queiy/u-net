import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import onnx
from typing import Any

# =================================================================
#                     !!! TODO 1: 关键参数设置 !!!
# =================================================================
# 请根据您训练时的实际参数和文件路径进行设置

MODEL_BACKBONE = 'resnet50'  # 骨干网络（固定为 resnet50）
NUM_CLASSES = 2  # 您的分割类别数
WEIGHTS_PATH = r'F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\logs\loss_2025_07_25_17_34_29\best_epoch_weights.pth'  # <<-- 替换为您的模型权重文件路径！
ONNX_OUTPUT_PATH = 'unet_exported_batchsize16.onnx'

# -------------------- 输入尺寸设置 --------------------
BATCH_SIZE = 4
INPUT_CHANNELS = 3  # RGB图像
INPUT_HEIGHT = 512  # <<-- 替换为您训练时使用的输入高度！
INPUT_WIDTH = 512  # <<-- 替换为您训练时使用的输入宽度！
USE_DEFORM = True # <--- 新增或修改此行！


# =================================================================


# =================================================================
#                      模型依赖模块定义 (CBAM)
# =================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(in_planes, ratio)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


# =================================================================
#                       模型依赖模块定义 (ResNet)
# =================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # 这里需要确保 model_data 路径和下载逻辑正确
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)

    del model.avgpool
    del model.fc
    return model


# =================================================================
#                 模型依赖模块定义 (DenseASPP 及其组件)
# =================================================================
class SEModule(nn.Module):
    def __init__(self, channels: int, ratio: int = 4) -> None:
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Any) -> Any:
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DeformConv2d(nn.Module):
    """
    警告: 此处为自定义实现的 DeformConv2d，ONNX 导出可能失败。
    如果导出失败，请检查此模块的张量操作是否兼容 ONNX 追踪。
    """

    def __init__(self, inc, outc, stride, kernel_size=3, padding=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)

        # 移除 register_backward_hook，因为它在推理时无用且可能干扰 ONNX 导出
        # self.p_conv.register_backward_hook(self._set_lr)

        self.bn = nn.BatchNorm2d(outc)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # 移除 register_backward_hook
            # self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # 移除此静态方法的使用，ONNX 导出时不需要
        pass

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        # 核心张量操作（ONNX 风险点）
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        out = self.bn(out)
        out = F.relu(out)
        return out

    # 以下辅助函数也可能引起 ONNX 导出问题
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, device=self.conv.weight.device),
            # 添加 device
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1,
                         device=self.conv.weight.device))  # 添加 device
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride, device=self.conv.weight.device),  # 添加 device
            torch.arange(1, w * self.stride + 1, self.stride, device=self.conv.weight.device))  # 添加 device
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset


import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPoolModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StripPoolModule, self).__init__()

        # 强制使用静态 AvgPool2d 模拟 Strip Pooling。
        # 假设输入特征图尺寸为 H x W (例如 16x16)

        # H方向的池化 (Pool to H x 1)：使用 kernel_size = (1, W)
        # 我们使用 Global Pooling 替代：将整个宽度压缩到 1
        # 在 16x16 的特征图上，宽度 W=16。
        self.strip_pool_h_op = nn.AvgPool2d(kernel_size=(1, 16), stride=1)
        self.conv1x1_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # W方向的池化 (Pool to 1 x W)：使用 kernel_size = (H, 1)
        # 我们使用 Global Pooling 替代：将整个高度压缩到 1
        # 在 16x16 的特征图上，高度 H=16。
        self.strip_pool_w_op = nn.AvgPool2d(kernel_size=(16, 1), stride=1)
        self.conv1x1_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        dilation_rates = [1, 6, 12, 18]
        self.dilated_convs_h = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.dilated_convs_w = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.final_conv1x1 = nn.Conv2d(out_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 注意: nn.AvgPool2d(kernel_size=(H, W)) 的输出尺寸是 (H_in - H_k + 1) x (W_in - W_k + 1)
        # 对于 16x16 输入:
        # strip_pool_h_op: (16 - 1 + 1) x (16 - 16 + 1) = 16 x 1
        # strip_pool_w_op: (16 - 16 + 1) x (16 - 1 + 1) = 1 x 16

        strip_h = self.strip_pool_h_op(x)
        strip_h = self.conv1x1_h(strip_h)
        # F.interpolate 仍然使用 x.size()[2:]，这是可接受的 ONNX::Resize 操作。
        strip_h = F.interpolate(strip_h, size=x.size()[2:], mode='bilinear', align_corners=False)

        strip_w = self.strip_pool_w_op(x)
        strip_w = self.conv1x1_w(strip_w)
        strip_w = F.interpolate(strip_w, size=x.size()[2:], mode='bilinear', align_corners=False)

        dilated_h = sum(conv(strip_h) for conv in self.dilated_convs_h)
        dilated_w = sum(conv(strip_w) for conv in self.dilated_convs_w)
        combined = torch.cat([dilated_h, dilated_w], dim=1)
        out = self.final_conv1x1(combined)
        out = self.sigmoid(out)
        out = x * out
        out = self.reduce(out)
        return out

class DenseAsppModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: list[int], use_deform: bool = False) -> None:
        super(DenseAsppModule, self).__init__()
        self.use_deform = use_deform
        self.dilations = dilations

        if use_deform:
            self.conv1x1 = nn.Sequential(
                DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.at_conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[0], padding=dilations[0],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
        )
        self.at_conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[1], padding=dilations[1],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
        )
        self.at_conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[2], padding=dilations[2],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
        )
        self.at_conv24 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[3], padding=dilations[3],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
        )

        self.proj12 = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj18 = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj24 = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.strip_pool = StripPoolModule(in_channels, out_channels)
        total_branches = 1 + len(dilations) + 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * total_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.squeeze_excite = SEModule(channels=out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1x1(x)
        atrous_outputs = []

        conv6 = self.at_conv6(x)
        atrous_outputs.append(conv6)

        x2 = self.proj12(torch.cat((x, conv6), dim=1))
        conv12 = self.at_conv12(x2)
        atrous_outputs.append(conv12)

        x3 = self.proj18(torch.cat((x2, conv12), dim=1))
        conv18 = self.at_conv18(x3)
        atrous_outputs.append(conv18)

        x4 = self.proj24(torch.cat((x3, conv18), dim=1))
        conv24 = self.at_conv24(x4)
        atrous_outputs.append(conv24)

        strip_pool = self.strip_pool(x)
        strip_pool = F.interpolate(strip_pool, size=x1.size()[2:], mode='bilinear', align_corners=False)

        combined_output = torch.cat((x1, *atrous_outputs, strip_pool), dim=1)

        aspp_output = self.final_conv(combined_output)
        aspp_output = self.batch_norm(aspp_output)
        aspp_output = self.relu(aspp_output)
        aspp_output = self.squeeze_excite(aspp_output)

        return aspp_output


# =================================================================
#                         Unet 的 Up-Sampling 模块
# =================================================================
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        # 确保使用 nn.UpsamplingBilinear2d 或 F.interpolate，ONNX 兼容性好
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # inputs1 是 skip connection, inputs2 是上采样的特征
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


# =================================================================
#                           Unet 主模型
# =================================================================
class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            # self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            # 添加降维卷积层，将2048通道降至1024通道
            self.downsample = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(1024)
            self.relu = nn.ReLU(inplace=True)

            # 保持DenseASPP的输入通道数为1024不变
            self.denseaspp = DenseAsppModule(in_channels=1024, out_channels=256,
                                             dilations=[4, 8, 12, 16], use_deform=USE_DEFORM)
            # 调整后续通道数计算
            in_filters = [192, 512, 1024, 1024 + 256]  # feat4 (1024) + denseaspp (256)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(1280, out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone
        # 添加 skip connection 的 CBAM 模块
        self.cbam1 = CBAM(64)  # 对应 feat1
        self.cbam2 = CBAM(256)  # 对应 feat2
        self.cbam3 = CBAM(512)  # 对应 feat3
        self.cbam4 = CBAM(1024)  # 对应 feat4
        self.cbam5 = CBAM(256)  # ASPP 输出通道（如果是 256）

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
            feat5 = self.denseaspp(feat5)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

            # print("feat1.shape,feat2.shape,feat3.shape,feat4.shape,feat5.shape 输入cbam前尺寸与通道数", feat1.shape, feat2.shape, feat3.shape, feat4.shape, feat5.shape)
            feat1 = self.cbam1(feat1)
            # print("feat1.shape", feat1.shape)
            feat2 = self.cbam2(feat2)
            # print("feat2.shape",feat2.shape)
            feat3 = self.cbam3(feat3)
            # print("feat3.shape", feat3.shape)
            feat4 = self.cbam4(feat4)
            # print("feat4.shape", feat4.shape)

            # 应用降维层
            feat5 = self.downsample(feat5)
            feat5 = self.bn(feat5)
            feat5 = self.relu(feat5)
            # print("feat5.shape 输入denseaspp前尺寸与通道数", feat5.shape)
            feat5 = self.denseaspp(feat5)
            # print("feat5.shape  经过denseaspp后输出尺寸与通道数", feat5.shape)

            feat5 = self.cbam5(feat5)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

# =================================================================
#                       ONNX 导出执行函数
# =================================================================

def convert_pytorch_to_onnx():
    print(f"--- 1. 实例化和加载 PyTorch 模型 (Backbone: {MODEL_BACKBONE}) ---")

    model = Unet(num_classes=NUM_CLASSES, pretrained=False, backbone=MODEL_BACKBONE)

    # 加载预训练权重
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
        # 尝试加载 state_dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        model.eval()
        print("模型加载成功并设置为评估模式。")

    except FileNotFoundError:
        print(f"致命错误: 权重文件未找到! 请检查路径: {WEIGHTS_PATH}")
        return
    except Exception as e:
        print(f"加载权重失败，请检查权重路径、state_dict 键或模型结构是否匹配: {e}")
        return

    print("--- 2. 准备示例输入张量 ---")
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                              dtype=torch.float32).to(model.final.weight.device)  # 确保设备一致
    print(f"示例输入形状: {dummy_input.shape}")

    # 定义动态轴 (允许运行时 Batch Size 变化)
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    print("--- 3. 执行 ONNX 导出 ---")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=17,  # 推荐使用较新的版本
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            # optimize_for_inference=True,
        )
        print(f"\n✅ 转换完成！ONNX 模型已保存到: {ONNX_OUTPUT_PATH}")

        # ONNX 文件检查
        onnx_model = onnx.load(ONNX_OUTPUT_PATH)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 文件结构检查通过！")

    except Exception as e:
        print(f"\n❌ ONNX 导出失败! 错误信息: {e}")
        print("---------------------------------------------------------")
        print("常见失败原因：自定义实现的 DeformConv2d 中使用了 ONNX 不支持的操作。")
        print("解决方案：尝试将 `DenseAsppModule` 中的 `use_deform` 设为 `False`，或替换 DeformConv2d 为 ONNX 兼容版本。")
        print("---------------------------------------------------------")


if __name__ == '__main__':
    convert_pytorch_to_onnx()