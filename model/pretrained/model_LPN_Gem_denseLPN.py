import argparse
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import timm


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x

def block_n(block):#！！
    if block == 4:
        block_n = 10
    elif block == 3:
        block_n = 6
    elif block == 2:
        block_n = 3
    return block_n

class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, x_pre=None, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1) # torch.Size([2, 8, 8, 2048])
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1) # torch.Size([2, 2048, 8, 8])
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1))) # torch.Size([2, 2048, 1, 1])
        x = x.view(x.size(0), x.size(1)) # torch.Size([2, 2048])
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(ResidualBlock, self).__init__()
        out_channels=in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # If the input and output dimensions are not the same, adjust them using a 1x1 convolution
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels=32):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.layer1 = ResidualBlock(in_channels=in_channels)
        self.layer2 = ResidualBlock(in_channels=in_channels)
        self.layer3 = ResidualBlock(in_channels=in_channels)
        self.layer4 = ResidualBlock(in_channels=in_channels)
        self.layer5 = ResidualBlock(in_channels=in_channels)

    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out



"""
我们基于上面的官方模型，进行适当修改。
"""
# Define the ResNet50-based part Model
class ft_net_resnet(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_resnet, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            self.model = model_ft
            # self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.model = model_ft
            # self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            self.model = model_ft

        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool
            # self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # torch.Size([2, 2048, 32, 32])
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        # x = self.classifier(x)
        return x


class ft_net_resnet_lpn(nn.Module):
    def __init__(self, class_num, droprate=0.5, decouple=False, block=4,stride=1):
        super(ft_net_resnet_lpn, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        self.pool='avg'
        self.model = model_ft
        self.decouple = decouple
        self.block = block
        self.bn = nn.BatchNorm1d(2048, affine=False)

        # Create a GeM layer for each block
        # 常规的初始化 Gem 或 Gem_new1
        for i in range(self.block):
            gem = 'gem' + str(i)
            setattr(self, gem, GeM(2048))


    def forward(self, x):
        # version 1
        # x = self.model.forward_features(x)
        # version 2
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)  # torch.Size([2, 1024, 32, 32])
        x = self.model.layer4(x)  # torch.Size([2, 2048, 32, 32])
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x,pool1_gem=False) # ！！！需要替换这里
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
        return x

    def get_part_pool(self, x, pool='avg', pool1_gem=False, no_overlap=True):
        result = []
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)#4
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)

        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                x_pre = None
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                if pool1_gem==True:
                    # res = self.gem_layers[i-1](x_curr, x_pre)
                    name = 'gem' + str(i - 1)
                    gem = getattr(self, name)
                    res = gem.gem(x_curr, x_pre)
                else:
                    res = self.avg_pool(x_curr, x_pre)
                result.append(res)

            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                if pool1_gem==True:
                    # res = self.gem_layers[i-1](x, x_pre)
                    name = 'gem' + str(i - 1)
                    gem = getattr(self, name)
                    res = gem.gem(x, x_pre)
                else:
                    res = self.avg_pool(x, x_pre)
                result.append(res)
        return torch.stack(result, dim=2)

    def get_part_pool_shift_new3(self, x, pool='avg', pool1_gem=False, no_overlap=True):
        # 软融合需要， 使用的时候需要解除注释
        shift_p = self.shift_blocks(x).unsqueeze(-1).unsqueeze(-1) # torch.Size([2, 3, 1, 1])

        result_temp = [[], [], []]
        H, W = x.size(2), x.size(3) # torch.Size([2, 2048, 32, 32])
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)#4
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)

        shift_h=[-2, 0, 2]
        shift_w=[-2, 0, 2]

        for j in range(len(shift_h)):
            for i in range(self.block):
                i = i + 1
                if i < self.block:
                    x_curr = x[:, :, (c_h - i * per_h + shift_h[j]):(c_h + i * per_h + shift_h[j]),
                             (c_w - i * per_w + shift_w[j]):(c_w + i * per_w + shift_w[j])]  # 取实心分区，第一块开始就引入偏移量
                    x_pre = None
                    if no_overlap and i > 1:
                        x_pre = x[:, :, (c_h - (i - 1) * per_h + shift_h[j]):(c_h + (i - 1) * per_h + shift_h[j]),
                                (c_w - (i - 1) * per_w + shift_w[j]):(c_w + (i - 1) * per_w + shift_w[j])]
                        x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                        x_curr = x_curr - x_pad
                    if pool1_gem == True:
                        # res = self.gem_layers[i - 1](x_curr, x_pre)
                        name = 'gem' + str(i - 1)
                        gem = getattr(self, name)
                        res = gem(x_curr, x_pre)
                    else:
                        res = self.avg_pool(x_curr, x_pre)
                    result_temp[j].append(res)

                else:
                    if no_overlap and i > 1:
                        x_pre = x[:, :, (c_h - (i - 1) * per_h + shift_h[j]):(c_h + (i - 1) * per_h + shift_h[j]),
                                (c_w - (i - 1) * per_w + shift_w[j]):(c_w + (i - 1) * per_w + shift_w[j])]
                        pad_h = c_h - (i - 1) * per_h
                        pad_w = c_w - (i - 1) * per_w
                        if x_pre.size(2) + 2 * pad_h == H:
                            x_pad = F.pad(x_pre, (pad_w + shift_w[j], pad_w - shift_w[j], pad_h + shift_h[j], pad_h - shift_h[j]),"constant", 0)
                        else:
                            ep = H - (x_pre.size(2) + 2 * pad_h)
                            x_pad = F.pad(x_pre, (pad_w + ep + shift_w[j], pad_w - shift_w[j], pad_h + ep + shift_h[j], pad_h - shift_h[j]), "constant",0)
                        x_out = x - x_pad
                    if pool1_gem == True:
                        # res = self.gem_layers[i - 1](x_out, x_pre)
                        name = 'gem' + str(i - 1)
                        gem = getattr(self, name)
                        res = gem(x_out, x_pre)
                    else:
                        res = self.avg_pool(x_out, x_pre)
                    result_temp[j].append(res)

        # 对角线偏移的三种分区。右上，中间，左下
        result_0 = torch.stack(result_temp[0], dim=2) # torch.Size([2, 2048, 4])
        result_1 = torch.stack(result_temp[1], dim=2)
        result_2 = torch.stack(result_temp[2], dim=2)

        # 软融合
        result = torch.cat((result_0.unsqueeze(1),result_1.unsqueeze(1),result_2.unsqueeze(1)),dim=1)
        result = torch.sum((result*shift_p),dim=1) # 使用的时候需要解除注释

        return result # torch.Size([2, 2048, 4])

    def avg_pool(self, x_curr, x_pre=None):
        h, w = x_curr.size(2), x_curr.size(3)
        if x_pre == None:
            h_pre = w_pre = 0.0
        else:
            h_pre, w_pre = x_pre.size(2), x_pre.size(3)
        pix_num = h * w - h_pre * w_pre
        avg = x_curr.flatten(start_dim=2).sum(dim=2).div_(pix_num)
        return avg

class ft_net_resnet_dense_lpn_shift3_new1(nn.Module):
    def __init__(self, class_num, droprate=0.5, decouple=False, block=4,stride=1):
        super(ft_net_resnet_dense_lpn_shift3_new1, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        self.pool='avg'
        self.model = model_ft
        self.decouple = decouple
        self.block = block
        self.bn = nn.BatchNorm1d(2048, affine=False)#1024->2048

        # 常规的初始化 Gem 或 Gem_new1
        for i in range(10):
            gem = 'gem' + str(i)
            setattr(self, gem, GeM(2048))

        # Create shift-guided parameters， 偏移分区软融合的时候需要初始化网络参数。 使用 get_part_pool_shift_new3 的时候需要解除注释
        self.shift_blocks = nn.Sequential(torch.nn.Conv2d(2048, 1024, kernel_size=1), # stage4的channels, 可以尝试换成maxpooling和avgpooling处理通道维度
                                          torch.nn.ReLU(),
                                          torch.nn.AdaptiveAvgPool2d((1, 1)), # 1 1024 1 1
                                          torch.nn.Flatten(),
                                          torch.nn.Linear(1024, 512), # H * W！！
                                          torch.nn.Dropout(0.5),
                                          torch.nn.Linear(512, 512),
                                          torch.nn.Dropout(0.5),
                                          torch.nn.Linear(512, 3), # 三块进行softmax
                                          torch.nn.Softmax(dim=1)
                                          )

    def forward(self, x):
        # version 1
        # x = self.model.forward_features(x)
        # version 2
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)  # torch.Size([2, 1024, 32, 32])
        x = self.model.layer4(x)  # torch.Size([2, 2048, 16, 16])
        if self.pool == 'avg+max':
            x1 = self.get_part_pool_dense(x, pool='avg')
            x2 = self.get_part_pool_dense(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool_dense_shift_new3(x,pool1_gem=True) # ！！！需要替换这里
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool_dense(x)
            x = x.view(x.size(0), x.size(1), -1)
        return x

    def get_part_pool_dense(self, x, block=4, pool='avg', pool1_gem=False):
        result = []
        base_feature = []
        pixels_n = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1))
        H, W = x.size(2), x.size(3)  # 计算input的宽高
        c_h, c_w = int(H / 2), int(W / 2)  # 计算中心点距离
        per_h, per_w = H / (2 * block), W / (2 * block)  # 根据block的数量，等距计算1个block的宽高
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (block - c_h) * 2, W + (block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear',
                                          align_corners=True)  # 如果图片尺寸过小，需要插值放大。
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * block), W / (2 * block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整

        for i in range(block):
            i = i + 1
            if i < block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]  # 由中心向外扩张
                x_pre = None
                if i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)  # 扩大一圈，全部用0填充。
                    x_curr = x_curr - x_pad
                base_feature.append(x_curr)
            else:
                if i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                base_feature.append(x)

        x_0 = base_feature[0]
        x_1 = base_feature[1]
        x_2 = base_feature[2]
        x_3 = base_feature[3]

        x_4 = x_1 + F.pad(x_0, (per_h, per_h, per_w, per_w), "constant", 0)  # 第二阶段
        x_5 = x_2 + F.pad(x_1, (per_h, per_h, per_w, per_w), "constant", 0)
        x_6 = x_3 + F.pad(x_2, (per_h, per_h, per_w, per_w), "constant", 0)

        x_7 = x_2 + F.pad(x_4, (per_h, per_h, per_w, per_w), "constant", 0)  # 第三阶段
        x_8 = x_3 + F.pad(x_5, (per_h, per_h, per_w, per_w), "constant", 0)

        # x_9 = x_3 + F.pad(x_7, (per_h, per_h, per_w, per_w), "constant", 0)  # 第四阶段
        x_9 = x
        base_feature.append(x_4)
        base_feature.append(x_5)
        base_feature.append(x_6)
        base_feature.append(x_7)
        base_feature.append(x_8)
        base_feature.append(x_9)

        pixel_0 = x_0.size(2) * x_0.size(3)
        pixel_1 = x_1.size(2) * x_1.size(3) - x_0.size(2) * x_0.size(3)
        pixel_2 = x_2.size(2) * x_2.size(3) - x_1.size(2) * x_1.size(3)
        pixel_3 = x_3.size(2) * x_3.size(3) - x_2.size(2) * x_2.size(3)
        pixel_4 = pixel_0 + pixel_1
        pixel_5 = pixel_1 + pixel_2
        pixel_6 = pixel_2 + pixel_3
        pixel_7 = pixel_2 + pixel_4
        pixel_8 = pixel_3 + pixel_5
        pixel_9 = pixel_7 + pixel_3
        pixels_n.append(pixel_0)
        pixels_n.append(pixel_1)
        pixels_n.append(pixel_2)
        pixels_n.append(pixel_3)
        pixels_n.append(pixel_4)
        pixels_n.append(pixel_5)
        pixels_n.append(pixel_6)
        pixels_n.append(pixel_7)
        pixels_n.append(pixel_8)
        pixels_n.append(pixel_9)
        # for i in range(10):
        #     pixels_n.append(eval(f'pixel_{i}'))
        for j in range(len(base_feature)):
            x_curr = base_feature[j]
            if j==0:
                x_pre = None
            else:
                x_pre = base_feature[j - 1]

            if pool1_gem == True:
                # res = self.gem_layers[i-1](x_curr, x_pre)
                name = 'gem' + str(i - 1)
                gem = getattr(self, name)
                # res = gem.gem(x_curr, x_pre,pix_num=pixels_n[j]) #Gem_new1_dense
                res = gem.gem(x_curr, x_pre) #！！Gem
                result.append(res)
            else:
                if j < 1:
                    avgpool = self.avg_pool(x_curr, pixels_n[j])
                else:
                    x_pre = base_feature[j - 1]
                    avgpool = self.avg_pool(x_curr, pixels_n[j])
                result.append(avgpool)
        return torch.stack(result, dim=2)


    def get_part_pool_dense_shift_new3(self, x, block=4, pool='avg', pool1_gem=False):
        # 软融合需要， 使用的时候需要解除注释
        shift_p = self.shift_blocks(x).unsqueeze(-1).unsqueeze(-1) # torch.Size([2, 3, 1, 1])
        pixels_n = []
        result_temp = [[], [], []]
        base_feature = []
        # if pool == 'avg':
        #     pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # elif pool == 'max':
        #     pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)  # 计算input的宽高
        c_h, c_w = int(H / 2), int(W / 2)  # 计算中心点距离
        per_h, per_w = H / (2 * block), W / (2 * block)  # 根据block的数量，等距计算1个block的宽高
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (block - c_h) * 2, W + (block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear',
                                          align_corners=True)  # 如果图片尺寸过小，需要插值放大。
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * block), W / (2 * block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整

        shift_h = [-2, 0, 2]
        shift_w = [-2, 0, 2]
        for j in range(len(shift_h)):
            base_feature = []
            for i in range(block):
                i = i + 1
                if i < block:
                    x_curr = x[:, :, (c_h - i * per_h + shift_h[j]):(c_h + i * per_h + shift_h[j]), (c_w - i * per_w + shift_w[j]):(c_w + i * per_w + shift_w[j])]  # 由中心向外扩张
                    x_pre = None
                    if i > 1:
                        x_pre = x[:, :, (c_h - (i - 1) * per_h + shift_h[j]):(c_h + (i - 1) * per_h + shift_h[j]),
                                (c_w - (i - 1) * per_w + shift_w[j]):(c_w + (i - 1) * per_w + shift_w[j])]
                        x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)  # 扩大一圈，全部用0填充。
                        x_curr = x_curr - x_pad
                    base_feature.append(x_curr)
                else:
                    if i > 1:
                        x_pre = x[:, :, (c_h - (i - 1) * per_h + shift_h[j]):(c_h + (i - 1) * per_h + shift_h[j]),
                                (c_w - (i - 1) * per_w + shift_w[j]):(c_w + (i - 1) * per_w + shift_w[j])]
                        pad_h = c_h - (i - 1) * per_h
                        pad_w = c_w - (i - 1) * per_w
                        # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                        if x_pre.size(2) + 2 * pad_h == H:
                            x_pad = F.pad(x_pre, (pad_w + shift_w[j], pad_w - shift_w[j], pad_h + shift_h[j], pad_h - shift_h[j]),"constant", 0)
                        else:
                            ep = H - (x_pre.size(2) + 2 * pad_h)
                            x_pad = F.pad(x_pre, (pad_w + ep + shift_w[j], pad_w - shift_w[j], pad_h + ep + shift_h[j], pad_h - shift_h[j]), "constant",0)
                        x = x - x_pad
                    base_feature.append(x)

            x_0 = base_feature[0]
            x_1 = base_feature[1]
            x_2 = base_feature[2]
            x_3 = base_feature[3]

            x_4 = x_1 + F.pad(x_0, (per_h, per_h, per_w, per_w), "constant", 0)  # 第二阶段
            x_5 = x_2 + F.pad(x_1, (per_h, per_h, per_w, per_w), "constant", 0)
            x_6 = x_3 + F.pad(x_2, (per_h, per_h, per_w, per_w), "constant", 0)

            x_7 = x_2 + F.pad(x_4, (per_h, per_h, per_w, per_w), "constant", 0)  # 第三阶段
            x_8 = x_3 + F.pad(x_5, (per_h, per_h, per_w, per_w), "constant", 0)

            # x_9 = x_3 + F.pad(x_7, (per_h, per_h, per_w, per_w), "constant", 0)  # 第四阶段
            x_9 = x
            base_feature.append(x_4)
            base_feature.append(x_5)
            base_feature.append(x_6)
            base_feature.append(x_7)
            base_feature.append(x_8)
            base_feature.append(x_9)

            pixel_0 = x_0.size(2) * x_0.size(3)
            pixel_1 = x_1.size(2) * x_1.size(3) - x_0.size(2) * x_0.size(3)
            pixel_2 = x_2.size(2) * x_2.size(3) - x_1.size(2) * x_1.size(3)
            pixel_3 = x_3.size(2) * x_3.size(3) - x_2.size(2) * x_2.size(3)
            pixel_4 = pixel_0 + pixel_1
            pixel_5 = pixel_1 + pixel_2
            pixel_6 = pixel_2 + pixel_3
            pixel_7 = pixel_2 + pixel_4
            pixel_8 = pixel_3 + pixel_5
            pixel_9 = pixel_7 + pixel_3
            pixels_n.append(pixel_0)
            pixels_n.append(pixel_1)
            pixels_n.append(pixel_2)
            pixels_n.append(pixel_3)
            pixels_n.append(pixel_4)
            pixels_n.append(pixel_5)
            pixels_n.append(pixel_6)
            pixels_n.append(pixel_7)
            pixels_n.append(pixel_8)
            pixels_n.append(pixel_9)
            # for i in range(10):
            #     pixels_n.append(eval(f'pixel_{i}'))
            for k in range(len(base_feature)):
                x_curr = base_feature[k]
                if k == 0:
                    x_pre = None
                else:
                    x_pre = base_feature[k - 1]

                if pool1_gem == True:
                    # res = self.gem_layers[i-1](x_curr, x_pre)
                    name = 'gem' + str(k)
                    gem = getattr(self, name)
                    res = gem.gem(x_curr, x_pre)
                    result_temp[j].append(res)
                else:
                    if k < 1:
                        avgpool = self.avg_pool(x_curr, pixels_n[k])
                    else:
                        x_pre = base_feature[k]
                        avgpool = self.avg_pool(x_curr, pixels_n[k])
                    result_temp[j].append(avgpool)

        result_0 = torch.stack(result_temp[0], dim=2)
        result_1 = torch.stack(result_temp[1], dim=2)
        result_2 = torch.stack(result_temp[2], dim=2)
        # 软融合
        result = torch.cat((result_0.unsqueeze(1),result_1.unsqueeze(1),result_2.unsqueeze(1)),dim=1)
        result = torch.sum((result*shift_p),dim=1) # 使用的时候需要解除注释
        return result


    def avg_pool(self, x_curr, pixel_n=None):
        h, w = x_curr.size(2), x_curr.size(3)
        # pix_num = h * w - h_pre * w_pre
        pix_num = pixel_n
        avg = x_curr.flatten(start_dim=2).sum(dim=2).div_(pix_num)
        return avg


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=1, pool='avg', share_weight=False, LPN=False, dense_LPN=False ,
                 block=4, decouple=False, swin=False, resnet=False):
        super(two_view_net, self).__init__()
        self.resnet=resnet
        self.LPN = LPN
        self.dense_LPN = dense_LPN
        self.block = block
        self.decouple = decouple
        self.pool = pool
        self.sqr = True  # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        if resnet:
            # resnet50-based_model
            if LPN:
                self.model_1 = ft_net_resnet_lpn(class_num, droprate, decouple=decouple, block=block)
            elif dense_LPN:
                self.model_1 = ft_net_resnet_dense_lpn_shift3_new1(class_num, droprate, decouple=decouple, block=block)
            else:
                self.model_1 = ft_net_resnet(class_num, droprate)

        # 判断是否使用共享权重
        if share_weight:
            self.model_2 = self.model_1
        else:
            if resnet:
                # resnet50-based_model
                if LPN:
                    self.model_2 = ft_net_resnet_lpn(class_num, droprate, decouple=decouple, block=block)
                elif dense_LPN:
                    self.model_2 = ft_net_resnet_dense_lpn_shift3_new1(class_num, droprate, decouple=decouple, block=block)
                else:
                    self.model_2 = ft_net_resnet(class_num, droprate)

        if LPN or dense_LPN or self.pool == 'lpn':
            if resnet:
                if dense_LPN:
                    for i in range(block_n(self.block)):
                        name = 'classifier' + str(i)
                        setattr(self, name, ClassBlock(2048, class_num, droprate))##ClassBlock_dense_LPN
                else:
                    for i in range(self.block):
                        name = 'classifier' + str(i)
                        setattr(self, name, ClassBlock(2048, class_num, droprate))

        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool == 'avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)


    def forward(self, x1, x2):
        if self.LPN or self.dense_LPN or self.pool == 'lpn':
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1) # torch.Size([2, 2048, 4])
                if self.decouple:
                    y1 = self.part_classifier(x1[0], self.dense_LPN)
                else:
                    y1 = self.part_classifier(x1, self.dense_LPN)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                if self.decouple:
                    y2 = self.part_classifier(x2[0], self.dense_LPN)
                else:
                    y2 = self.part_classifier(x2, self.dense_LPN)

        else:
            if x1 is None:
                y1 = None
            else:
                # x1 = self.vgg1.features(x1)
                x1 = self.model_1(x1)
                if self.decouple:
                    y1 = self.classifier(x1[0])
                else:
                    y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                # x2 = self.vgg2.features(x2)
                x2 = self.model_2(x2)
                if self.decouple:
                    y2 = self.classifier(x2[0])
                else:
                    y2 = self.classifier(x2)
        if self.decouple:
            return [y1, x1[1]], [y2, x2[1]]
        return y1, y2

    def part_classifier(self, x, dense_LPN):
        part = {}
        predict = {}
        if dense_LPN :
            for i in range(x.shape[-1]):  # 原来的是self.block
                # part[i] = torch.squeeze(x[:,:,i])
                part[i] = x[:, :, i].view(x.size(0), -1)
                name = 'classifier' + str(i)
                c = getattr(self, name)
                predict[i] = c(part[i])
            y = []
            for i in range(x.shape[-1]):
                y.append(predict[i])
            if not self.training:
                return torch.stack(y, dim=2)
            return y
        else:
            for i in range(self.block):
                # part[i] = torch.squeeze(x[:,:,i])
                part[i] = x[:, :, i].view(x.size(0), -1)
                name = 'classifier' + str(i)
                c = getattr(self, name)
                predict[i] = c(part[i])
            y = []
            for i in range(self.block):
                y.append(predict[i])
            if not self.training:
                return torch.stack(y, dim=2)
            return y



'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = two_view_net(class_num=701, droprate=0.5, stride=1, share_weight=True, LPN=False, dense_LPN=True ,block=4,
                       swin=False, resnet=True)
    # net.eval()

    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    # net = ft_net(701)
    print(net)

    input = Variable(torch.FloatTensor(2, 3, 512, 512))
    output1, output2 = net(input, input)
    # output1,output2,output3 = net(input,input,input)
    # output1 = net(input)
    # print('net output size:')
    print(output1[0].shape)
    # print(output.shape)

    # for i in range(len(output1)):
    #     print(output1[i].shape)

    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)
