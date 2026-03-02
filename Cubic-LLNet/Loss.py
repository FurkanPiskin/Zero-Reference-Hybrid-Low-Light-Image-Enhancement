import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

        # 初始化权重参数
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(
            0)  # 在位置0处增加一个维度，再增加一个维度
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        # 权重参数赋值
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

        # 平均值池化
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_layer_subtraction(nn.Module):

    def __init__(self, patch_size):
        super(L_layer_subtraction, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, enhance_img_01, enhance_img_02, enhance_img_03, enhance_img_04):
        b, c, h, w = enhance_img_01.shape
        enhance_img_01 = torch.mean(enhance_img_01, 1, keepdim=True)
        enhance_img_02 = torch.mean(enhance_img_02, 1, keepdim=True)
        enhance_img_03 = torch.mean(enhance_img_03, 1, keepdim=True)
        enhance_img_04 = torch.mean(enhance_img_04, 1, keepdim=True)

        mean_12 = self.pool(enhance_img_01) - self.pool(enhance_img_02)
        mean_23 = self.pool(enhance_img_02) - self.pool(enhance_img_03)
        mean_34 = self.pool(enhance_img_03) - self.pool(enhance_img_04)

        d = torch.mean(torch.pow(mean_12, 2) + torch.pow(mean_23, 2) + torch.pow(mean_34, 2))
        return d


class L_ssim(nn.Module):

    def __init__(self, patch_size):
        super(L_ssim, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, enhance_img_01, enhance_img_02, enhance_img_03, enhance_img_04):
        b, c, h, w = enhance_img_01.shape
        enhance_img_01 = torch.mean(enhance_img_01, 1, keepdim=True)
        enhance_img_02 = torch.mean(enhance_img_02, 1, keepdim=True)
        enhance_img_03 = torch.mean(enhance_img_03, 1, keepdim=True)
        enhance_img_04 = torch.mean(enhance_img_04, 1, keepdim=True)

        mean_12 = self.pool(enhance_img_01) - self.pool(enhance_img_02)
        mean_23 = self.pool(enhance_img_02) - self.pool(enhance_img_03)
        mean_34 = self.pool(enhance_img_03) - self.pool(enhance_img_04)

        d = torch.mean(torch.pow(mean_12, 2) + torch.pow(mean_23, 2) + torch.pow(mean_34, 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, enhance_img_01, enhance_img_02, enhance_img_03, enhance_img_04):
        batch_size = enhance_img_01.size()[0]
        h_x = enhance_img_01.size()[2]
        w_x = enhance_img_01.size()[3]
        count_h = (enhance_img_01.size()[2] - 1) * enhance_img_01.size()[3]
        count_w = enhance_img_01.size()[2] * (enhance_img_01.size()[3] - 1)
        h_tv = torch.pow((enhance_img_01[:, :, 1:, :] - enhance_img_01[:, :, :h_x - 1, :]), 2).sum() + \
               torch.pow((enhance_img_02[:, :, 1:, :] - enhance_img_02[:, :, :h_x - 1, :]), 2).sum() + \
               torch.pow((enhance_img_03[:, :, 1:, :] - enhance_img_03[:, :, :h_x - 1, :]), 2).sum() + \
               torch.pow((enhance_img_04[:, :, 1:, :] - enhance_img_04[:, :, :h_x - 1, :]), 2).sum()

        w_tv = torch.pow((enhance_img_01[:, :, :, 1:] - enhance_img_01[:, :, :, :w_x - 1]), 2).sum() + \
               torch.pow((enhance_img_02[:, :, :, 1:] - enhance_img_02[:, :, :, :w_x - 1]), 2).sum() + \
               torch.pow((enhance_img_03[:, :, :, 1:] - enhance_img_03[:, :, :, :w_x - 1]), 2).sum() + \
               torch.pow((enhance_img_04[:, :, :, 1:] - enhance_img_04[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / (batch_size * 4)


class L_his(nn.Module):
    def __init__(self):
        super(L_his, self).__init__()

    def forward(self, enhance_img_1, enhance_img_2, enhance_img_3, enhance_img_4):
        b, c, h, w = enhance_img_1.shape
        enhance_image_sum = enhance_img_1 + enhance_img_2 + enhance_img_3 + enhance_img_4
        pixel_sum = torch.sum(enhance_image_sum, dim=(1, 2, 3))
        total_pixel = c * h * w
        pixel_mean = pixel_sum / total_pixel
        pixel_mean = pixel_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return pixel_mean
    
class ColorAngleLoss(nn.Module):
    def __init__(self):
        super(ColorAngleLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output, input_low):
        # Çıktı ve Girdi arasındaki renk vektörü açısı aynı kalmalı
        # Cosine Similarity 1'e yakın olmalı (Aynı yön)
        similarity = self.cos(output, input_low)
        # Loss: 1 - similarity (Benzerlik ne kadar azsa ceza o kadar büyük)
        loss = torch.mean(1 - similarity)
        return loss


class L_ContrastLoss(nn.Module):
    def __init__(self, block_size=4):
        super(L_ContrastLoss, self).__init__()
        self.block_size = block_size

    def forward(self, images):
        batch_size, num_channels, height, width = images.size()

        # 将图像划分为 4x4 的子块
        sub_blocks = images.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)

        # 获取子块的数量
        num_blocks_h = sub_blocks.size(2)
        num_blocks_w = sub_blocks.size(3)

        # 将子块展平为形状为 (batch_size, num_channels, num_blocks_h, num_blocks_w, block_size, block_size) 的张量
        sub_blocks_flat = sub_blocks.reshape(batch_size, num_channels, num_blocks_h, num_blocks_w, -1)

        # 计算子块内的最大值和最小值
        max_values, _ = torch.max(sub_blocks_flat, dim=4)
        min_values, _ = torch.min(sub_blocks_flat, dim=4)
        # print("max_value shape", max_values.shape)

        # 计算对比度数值
        s = (max_values - min_values) / (max_values + min_values + 1e-8)  # 避免除零错误
        # print("s shape", s.shape)

        # 计算所有子块的 s 值的和并除以子块的数量
        num_blocks = num_blocks_h * num_blocks_w * batch_size
        contrast_loss = torch.sum(s) / (num_blocks * batch_size)
        return contrast_loss


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    x1 = torch.rand((2, 3, 8, 8))
    x1 = x1.cuda()
    x2 = torch.ones((2, 3, 128, 128))
    x2 = x2.cuda()
    x3 = torch.randn((2, 3, 128, 128))
    x3 = x1.cuda()
    x4 = torch.randn((2, 3, 128, 128))
    x4 = x1.cuda()

    # L_spa = L_spa().cuda()
    # l_spa = L_spa(x1, x2)
    # print("l_spa size is", l_spa.shape)
    #
    # L_col = L_color().cuda()
    # l_col = L_col(x1)
    # print("l_col size is", l_col.shape)
    #
    # L_his = L_his().cuda()
    # l_his1= L_his(x1, x2, x3, x4)
    # print("L_his size is ", l_his1)

    # l_tv = L_TV().cuda()
    # L_tv1 = l_tv(x1, x2, x3, x4)
    # print("L_tv1 ", L_tv1)

    L_exp = L_exp(16, 0.6).cuda()
    l_exp = L_exp(x2)
    print("L_exp ", l_exp)
    print(x1)

    L_contrast = L_ContrastLoss(4).cuda()
    l_con = L_contrast(x1)
    print("L_contrast ", l_con)

    # l_layer = L_layer_subtraction(16).cuda()
    # l_layer_sub = l_layer(x1, x2, x3, x4)
    # print("l_layer_sub ", l_layer_sub)

