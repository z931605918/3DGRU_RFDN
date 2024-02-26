import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import nn


class SRCNN(nn.Module):
    def __init__(self, scale=4):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 2*(scale**2), kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)


        self.shuf = nn.PixelShuffle(scale)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.shuf(x)
        return x

if __name__ == "__main__":
    import time

    # 程序代码段运行
    net = SRCNN(scale=4)  #不同下采样因子
    net.cuda()
    # net.eval()
    in_ten = torch.randn((200, 2, 32, 32)).cuda()
    start = time.time()
    x = net(in_ten)
    end = time.time()
    #flops, params = profile(net.cuda(), inputs=(in_ten,))

    # print('计算量', flops)
    # print('参数量', params)
    print('时间',end - start)
