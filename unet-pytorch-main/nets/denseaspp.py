import torch
import torch.nn as nn
from typing import Any
import torch.nn.functional as F


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) 模块，用于通道注意力机制。
    该模块通过自适应地重新校准通道来提升特征表示能力。
    """
    def __init__(self, channels: int, ratio: int = 4) -> None:
        super(SEModule, self).__init__()

        # 自适应平均池化层，将输入特征图的每个通道的空间维度缩减到1
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 全连接层序列，用于Excitation操作
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),  # 降维操作，输出尺寸是输入通道数的1/ratio
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.Linear(channels // ratio, channels),  # 恢复到原始通道数
            nn.Sigmoid(),  # 使用Sigmoid激活函数，将输出限制在0到1之间
        )

    def forward(self, x: Any) -> Any:
        """
        前向传播函数，执行SE模块的Squeeze和Excitation操作。
        """
        b, c, _, _ = x.size()  # 获取输入张量的批次大小和通道数

        # Squeeze操作：使用自适应平均池化层
        y = self.avgpool(x).view(b, c)  # 将每个通道的特征图缩减到一个值

        # Excitation操作：使用全连接层序列
        y = self.fc(y).view(b, c, 1, 1)  # 重新调整张量形状

        # 返回经过通道注意力调整的特征图
        return x * y
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, stride, kernel_size=3, padding=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.bn = nn.BatchNorm2d(outc)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

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

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
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

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class StripPoolModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StripPoolModule, self).__init__()

        # Strip Pooling 1xW (水平条形池化)
        self.strip_pool_h = lambda x: F.adaptive_avg_pool2d(x, (x.size(2), 1))  # 高度不变，宽度变1
        self.conv1x1_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Strip Pooling Hx1 (垂直条形池化)
        self.strip_pool_w = lambda x: F.adaptive_avg_pool2d(x, (1, x.size(3)))  # 宽度不变，高度变1
        self.conv1x1_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 膨胀卷积块，带有多个膨胀率
        dilation_rates = [1, 6, 12, 18]
        self.dilated_convs_h = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.dilated_convs_w = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])

        # 最终的1x1卷积层和sigmoid激活函数
        self.final_conv1x1 = nn.Conv2d(out_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 水平条形池化
        strip_h = self.strip_pool_h(x)
        strip_h = self.conv1x1_h(strip_h)
        strip_h = F.interpolate(strip_h, size=x.size()[2:], mode='bilinear', align_corners=False)

        # 垂直条形池化
        strip_w = self.strip_pool_w(x)
        strip_w = self.conv1x1_w(strip_w)
        strip_w = F.interpolate(strip_w, size=x.size()[2:], mode='bilinear', align_corners=False)

        # 空洞卷积
        dilated_h = sum(conv(strip_h) for conv in self.dilated_convs_h)
        dilated_w = sum(conv(strip_w) for conv in self.dilated_convs_w)

        # 合并水平和垂直的特征
        combined = torch.cat([dilated_h, dilated_w], dim=1)

        # 最后的1x1卷积层和sigmoid激活
        out = self.final_conv1x1(combined)
        out = self.sigmoid(out)

        # 对输入进行加权
        out = x * out
        out = self.reduce(out)
        return out



class DenseAsppModule(nn.Module):
    """
    密集连接的空洞卷积模块
    """
    def __init__(self, in_channels: int, out_channels: int, dilations: list[int], use_deform: bool = False) -> None:
        super(DenseAsppModule, self).__init__()
        self.use_deform = use_deform
        self.dilations = dilations

        # 1x1 或 可变形卷积
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

        # 为每个膨胀率单独定义卷积分支
        self.at_conv6  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[0], padding=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
            # nn.Dropout2d(p=0.3)
        )
        self.at_conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[1], padding=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
            # nn.Dropout2d(p=0.25)
        )
        self.at_conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[2], padding=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
            # nn.Dropout2d(p=0.2)
        )
        self.at_conv24 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[3], padding=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEModule(channels=out_channels),
            # nn.Dropout2d(p=0.2)
        )

        # 输出和 x 拼接后，把通道数从 in+out 再投影回 in
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

        # 条纹池化模块
        self.strip_pool = StripPoolModule(in_channels, out_channels)

        # 最终融合卷积，输入通道数 = 1x1 + 4*空洞 + 条纹
        total_branches = 1 + len(dilations) + 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * total_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)  # 批归一化层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.squeeze_excite = SEModule(channels=out_channels)  # SE模块用于通道注意力机制
        self.dropout = nn.Dropout(p=0.2)  # Dropout层，用于防止过拟合

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.conv1x1(x)

        atrous_outputs = []


        #conv6
        conv6  = self.at_conv6(x)
        atrous_outputs.append(conv6)
        # conv12
        x2 = self.proj12(torch.cat((x, conv6), dim=1))
        conv12 = self.at_conv12(x2)
        atrous_outputs.append(conv12)
        # conv18
        x3 = self.proj18(torch.cat((x2, conv12), dim=1))
        conv18 = self.at_conv18(x3)
        atrous_outputs.append(conv18)
        # conv24
        x4 = self.proj24(torch.cat((x3, conv18), dim=1))
        conv24 = self.at_conv24(x4)
        atrous_outputs.append(conv24)

        # 替换全局平均池化层为StripPoolingModule
        strip_pool = self.strip_pool(x)

        # 确保所有张量具有相同的空间尺寸
        strip_pool = F.interpolate(strip_pool, size=x1.size()[2:], mode='bilinear', align_corners=False)

        # 合并所有输出特征图（1x1卷积输出、空洞卷积输出和条纹池化输出）
        combined_output = torch.cat((x1, *atrous_outputs, strip_pool), dim=1)

        # 最终的1x1卷积，用于生成DenseASPP模块的输出
        aspp_output = self.final_conv(combined_output)
        aspp_output = self.batch_norm(aspp_output)
        aspp_output = self.relu(aspp_output)
        aspp_output = self.squeeze_excite(aspp_output)

        return aspp_output