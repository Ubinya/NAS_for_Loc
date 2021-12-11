import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build_model_2d import AutoFeature, Disp
from models.build_model_3d import AutoMatching
import pdb
from time import time

# 最上层模型，定义feature网络和matching网络相关参数
class AutoStereo(nn.Module):
    def __init__(self, maxdisp=192, Fea_Layers=6, Fea_Filter=8, Fea_Block=4, Fea_Step=3, Mat_Layers=12, Mat_Filter=8, Mat_Block=4, Mat_Step=3):
        super(AutoStereo, self).__init__()
        self.maxdisp = maxdisp
        #define Feature parameters
        self.Fea_Layers = Fea_Layers
        self.Fea_Filter = Fea_Filter
        self.Fea_Block = Fea_Block
        self.Fea_Step = Fea_Step
        #define Matching parameters
        self.Mat_Layers = Mat_Layers
        self.Mat_Filter = Mat_Filter
        self.Mat_Block = Mat_Block
        self.Mat_Step = Mat_Step

# feature 和 matching
        self.feature  = AutoFeature(self.Fea_Layers, self.Fea_Filter, self.Fea_Block, self.Fea_Step)
        self.matching = AutoMatching(self.Mat_Layers, self.Mat_Filter, self.Mat_Block, self.Mat_Step)
        self.disp = Disp(self.maxdisp)

# 对module进行初始化，设置参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 定义forward函数
    def forward(self, x, y):
# 将两个图分别输入feature网络
        x = self.feature(x)
        y = self.feature(y)

# 创建一个新tensor名为cost，设置维度
        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_()
# 填充cost，一部分区域是x的feature，一部分区域是y的feature
        for i in range(int(self.maxdisp/3)):
            if i > 0 :
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
# 组合好的cost输入matching网络
        cost = self.matching(cost)
        disp0 = self.disp(cost)
        return disp0



