import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


def forward_warp2d(tenInput, tenFlow):
    backwarp_tenGrid = {}
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
    # end

    tenFlow = -torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                           tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)



def forward_warp3d(tenInput, tenFlow):
    backwarp_tenGrid = {}
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenDeepth= torch.linspace(-1.0,1.0, tenFlow.shape[2]).view(1,1,tenFlow.shape[2],1,1).expand(tenFlow.shape[0],-1,-1,tenFlow.shape[3],tenFlow.shape[4])
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3], 1).expand(tenFlow.shape[0], -1, tenFlow.shape[2],-1, tenFlow.shape[4])
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[4]).view(1, 1, 1, 1, tenFlow.shape[4]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], tenFlow.shape[3], -1)
        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([  tenHorizontal, tenVertical,tenDeepth  ], 1).cuda()
    # end
    tenFlow = -torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[4] - 1.0) / 2.0),
                          tenFlow[:, 1:2, :, :] / ((tenInput.shape[3] - 1.0) / 2.0) ,
                          tenFlow[:, 2:3, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample\
        (input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow)
         .permute(0, 2, 3, 4,1), mode='bilinear', padding_mode='zeros', align_corners=True)


# end
if __name__=='__main__':

    #test 2D
    x = torch.rand(5,6,5,5).cuda()
    tenFlow=torch.ones(5,2,5,5).cuda()
    y=forward_warp2d(x,tenFlow)
    plt.subplot(121)
    plt.imshow(x[0,0].cpu(),vmax=1,vmin=-1)
    plt.subplot(122)
    plt.imshow(y[0,0].cpu(),vmax=1,vmin=-1)
    plt.show()
    print(y)

    #test 3D
    x = torch.rand(1,1,5,5,5).cuda()
    tenFlow_1=torch.ones(1,2,5,5,5).cuda()
    tenFlow_2=torch.ones(1,1,5,5,5).cuda()
    tenFlow=torch.cat([tenFlow_1,tenFlow_2],dim=1)   # v,w
    y=forward_warp3d(x,tenFlow)
    plt.subplot(121)
    plt.imshow(x[0,0,0].cpu(),vmax=1,vmin=-1,cmap='jet')
    plt.subplot(122)
    plt.imshow(y[0,0,1].cpu(),vmax=1,vmin=-1,cmap='jet')
    plt.show()
    print(x)
    print(y)




