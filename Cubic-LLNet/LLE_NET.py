import torch
import torch.nn as nn
import torch.nn.functional as F
from Conv2dBlock import ConvBlock
from LLE_cubic import LLE_cubic
import math
# import pytorch_colors as colors
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv
import os
from thop import profile

class LLE_Net(nn.Module):
    def __init__(self):
        super(LLE_Net, self).__init__()
        self.iteration=4
        self.out_channels=self.iteration*3*2

        self.block1=ConvBlock(3,16) #3,32
        self.block2=ConvBlock(16,32) #32,64
        self.block3=ConvBlock(32,64) #64,128

        
        #self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        #self.flatten=nn.Flatten()
        #self.fc=nn.Linear(64,self.out_channels)
        self.final_conv = nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1=self.block1(x)
        x1=self.block2(x1)
        x1=self.block3(x1)
        #x1=self.avgpool(x1)
        #x1=self.flatten(x1)
        #x1=self.fc(x1)
        x1=self.final_conv(x1)
        x1=torch.sigmoid(x1)

        t1, b1, t2, b2, t3, b3, t4, b4 = torch.split(x1, 3, dim=1)
        enhance_image_1 = LLE_cubic.LLE_LUT(x, t1, b1)
        enhance_image_2 = LLE_cubic.LLE_LUT(enhance_image_1, t2, b2)
        enhance_image_3 = LLE_cubic.LLE_LUT(enhance_image_2, t3, b3)
        enhance_image_4 = LLE_cubic.LLE_LUT(enhance_image_3, t4, b4)
        params = torch.cat([t1, b1, t2, b2, t3, b3, t4, b4], dim=1)

        return enhance_image_1, enhance_image_2, enhance_image_3, enhance_image_4, params

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = LLE_Net().cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Parameters:", num_params)
    images=torch.randint(0,256,(1,3,900,600),dtype=torch.float32,requires_grad=True)
    print("*******************images*******************")
    input = images / 255.0
    input = input.cuda()
    le1, le2, le3, le4, params = model(input)
    print("output.shape is ", le4.shape)

    flops, params = profile(model, inputs=(input,))
    print(f"Estimated FLOPs: {flops}")
    print(f"Number of parameters: {params}")


        


