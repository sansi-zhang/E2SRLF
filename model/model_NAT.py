import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from torchvision.transforms import ToPILImage
import os


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        print("Using subpix-NAT model!")
        
        feaC = 16
        feaC_out = 8
        channel = 120
        channel_2 = 4
        # channel = 100
        mindisp, maxdisp = -2, 2
        
        self.scale_factor = cfg.scale_factor
        angRes = cfg.angRes
        self.angRes = angRes 
        self.Scale_target = int(1/cfg.scale_factor)

        self.init_feature = nn.Sequential(
            nn.Conv2d(1, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            nn.Conv2d(feaC,  feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC_out, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            )

        self.build_costvolume = BuildCostVolume(feaC_out, channel, angRes, mindisp, maxdisp, self.Scale_target)

        self.aggregation = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
        # 构建代价体
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            ResB3D(channel),
            ResB3D(channel),
            nn.Conv3d(channel, channel_2, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.aggregation2 = nn.Sequential(
            nn.Conv3d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel_2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel_2, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.super_regression = Regression(mindisp, maxdisp, self.Scale_target)
        
        # self.att = Attention(feaC_out)
        # self.att_D = Attention_depth(channel)


    def forward(self, x):
        x = SAI2MacPI(x, self.angRes)
        init_feat = self.init_feature(x)
        # init_feat,_ = self.att(init_feat)
        cost = self.build_costvolume(init_feat)
        # cost,_ = self.att_D(cost)
        
        cost = self.aggregation(cost)
        cost = rearrange(cost, 'b c (d s) h w -> b c d s h w', s=(self.Scale_target)**2).contiguous()
        # 将超分维度信息变成空间信息 (b,c,d,s,h,w) -> (b,c,d,H,W)
        cost = concatenate_blocks_4d(cost)
        cost = self.aggregation2(cost)

        super_disp = self.super_regression(cost.squeeze(dim=1))  
        
        ## 总的下采样
        super_disp_down = downsample_lf(super_disp, 1/self.Scale_target) / self.Scale_target
        
        
        return super_disp_down, super_disp

# 使用双三次插值方法进行下采样处理
def downsample_lf(lf, scale_factor):
    a1, a2, H, W = lf.shape
    lf_down_list = []
    for u in range(a1):
        for v in range(a2):
            lf_slice = lf[u, v]
            # 进行双三次插值下采样
            down_slice = F.interpolate(lf_slice.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='bicubic').squeeze()
            lf_down_list.append(down_slice)

    lf_down = torch.stack(lf_down_list).view(a1, a2, *down_slice.shape)
    return lf_down 


class BuildCostVolume(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp, scale):
        super(BuildCostVolume, self).__init__()
        self.DSAFE = nn.Conv2d(channel_in, channel_out, angRes, stride=angRes, padding=0, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.scale = scale

    def cal_pad_dila(self, d):
        if d < 0:
            dilat = int(abs(d) * self.angRes + 1)
            pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d))
        if d == 0:
            dilat = 1
            pad = 0
        if d > 0:
            dilat = int(abs(d) * self.angRes - 1)
            pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d) - self.angRes + 1)
        return dilat, pad

    def forward(self, x):
        cost_list = []
        Disp = []
    
        for d in torch.arange(self.mindisp, self.maxdisp + 0.5, step=0.5):  # Add fractional disparities
            for _ in range(self.scale**2):
                Disp.append(d.item())  # Ensure `d` is Python float

        for d in Disp:
            # Compute dilation and padding for both low and high integer disparities
            d_low = int(torch.floor(torch.tensor(d)))
            d_high = int(torch.ceil(torch.tensor(d)))
            w_low = d_high - d
            w_high = d - d_low
            
            dilat_low, pad_low = self.cal_pad_dila(d_low)
            dilat_high, pad_high = self.cal_pad_dila(d_high)
            
            # Apply convolution for both low and high disparities
            cost_low = F.conv2d(x, weight=self.DSAFE.weight, stride=self.angRes, dilation=dilat_low, padding=pad_low)
            cost_high = F.conv2d(x, weight=self.DSAFE.weight, stride=self.angRes, dilation=dilat_high, padding=pad_high)

            # Interpolate between low and high disparity costs
            cost = w_low * cost_low + w_high * cost_high
            cost_list.append(cost)
        
        # Stack cost along disparity dimension
        cost_volume = torch.stack(cost_list, dim=2)
        return cost_volume


class Regression(nn.Module):
    def __init__(self, mindisp, maxdisp, Scale_target):
        super(Regression, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.maxdisp = maxdisp * Scale_target
        self.mindisp = mindisp * Scale_target

    def forward(self, cost):
        score = self.softmax(cost)              # B, D, H, W
        temp = torch.zeros(score.shape).to(score.device)            # B, D, H, W
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = score[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)     # B, 1, H, W
        return disp


class SpaResB(nn.Module):
    def __init__(self, channels, angRes):
        super(SpaResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        # 使用空间注意力时需要对角度维度使用压缩求空间的平均
        self.pool = nn.AdaptiveAvgPool2d((1,1))    #[b,c,u,v,1,1]
        self.up = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.down = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B,C,H,W = x.shape
        avg = self.pool(x)
        up = self.up(avg)
        down = self.down(up)
        attention = self.sigmoid(down)
        output = attention*x
        output = output + x
        return output, attention   

class Attention_depth(nn.Module):
    def __init__(self, channel):
        super(Attention_depth, self).__init__()
        # 使用空间注意力时需要对角度维度使用压缩求空间的平均
        self.pool = nn.AdaptiveAvgPool1d(1)    #[b,c,u,v,1,1]
        self.up = nn.Sequential(
            nn.Conv3d(channel, channel*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.down = nn.Conv3d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = input
        x = x.permute(0, 1, 3, 4, 2)
        b,c,h,w,d = x.shape
        x = x.reshape(b, -1, d)
        avg = self.pool(x)
        avg = avg.reshape(b,c,h,w,1)
        avg = avg.permute(0, 1, 4, 2, 3)
        up = self.up(avg)
        down = self.down(up)
        attention = self.sigmoid(down)
        output = attention*input
        output = output + input
        return output, attention  

    

def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out

def concatenate_blocks_4d(input_tensor):
    
    # 获取输入的形状信息
    n, c, d, s, h, w = input_tensor.shape
    # 将每组的 copy_num 个通道进行拼接
    cent = int(math.sqrt(s))
    
    tensor = torch.zeros(n, c, d, cent, cent, h, w).to(input_tensor.device)
    
    for i in range(s):
        sy = i // cent
        sx = i % cent
        tensor[:,:,:,sy,sx,:,:] = input_tensor[:,:,:,i,:,:]
    tensor = rearrange(tensor, 'b c d s1 s2 h w -> b c d (s1 h) (s2 w)', s1=cent, s2=cent).contiguous()
    b, c, d, hs, ws = tensor.shape
    h, w = hs // cent, ws // cent
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(tensor[:, :, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=4))
    out = torch.cat(tempU, dim=3)
    
    return out 


if __name__ == "__main__":
    net = Net(angRes=9).cuda()
    from thop import profile
    input = torch.randn(1, 1, 576, 576).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops/ 1e9))
