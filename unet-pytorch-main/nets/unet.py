import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.denseaspp import DenseAsppModule
from nets.cbam import CBAM


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            # 添加降维卷积层，将2048通道降至1024通道
            self.downsample = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(1024)
            self.relu = nn.ReLU(inplace=True)


            # 保持DenseASPP的输入通道数为1024不变
            self.denseaspp = DenseAsppModule(in_channels=1024, out_channels=256,
                                             dilations=[4, 8, 12, 16], use_deform=True)
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
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
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
