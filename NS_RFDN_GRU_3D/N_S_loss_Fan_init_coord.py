import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import statistics

def forward_warp2d(tenInput, tenFlow):
    backwarp_tenGrid = {}
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
    # end

    tenFlow = -torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                           tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow)
    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border', align_corners=True)
# end

# x_g=torch.tensor([0,1.5,2,2.5,4]).cuda().float()
# y_g=torch.tensor([0,1,2,3,4]).cuda().float()
# Y_g,X_g=torch.meshgrid([y_g,x_g])
#
# x_new=torch.tensor([0,1,2,3,4]).cuda().float()
# y_new=torch.tensor([0,1,2,3,4]).cuda().float()
# Y_n,X_n=torch.meshgrid([y_new,x_new])
#
# Zn=(X_g).unsqueeze(dim=0).unsqueeze(dim=0)
#
# tenFlow=torch.stack([X_g-X_n,Y_g-Y_n],dim=0).unsqueeze(dim=0)
# out=forward_warp2d(Zn,tenFlow)
# plt.subplot(121)
# plt.imshow(Zn[0,0].cpu(),vmax=5,vmin=0,cmap='jet')
# plt.subplot(122)
# plt.imshow(out[0,0].cpu(),vmax=5,vmin=0,cmap='jet')
# plt.show()
# print(y)




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

# x = torch.rand(1,1,5,5,5).cuda()
# tenFlow_1=torch.ones(1,2,5,5,5).cuda()
# tenFlow_2=torch.ones(1,1,5,5,5).cuda()
# tenFlow=torch.cat([tenFlow_1,tenFlow_2],dim=1)   # u,v,w
# y=forward_warp3d(x,tenFlow)
# plt.subplot(121)
# plt.imshow(x[0,0,0].cpu(),vmax=1,vmin=-1,cmap='jet')
# plt.subplot(122)
# plt.imshow(y[0,0,1].cpu(),vmax=1,vmin=-1,cmap='jet')
# plt.show()
# print(x)
# print(y)


def cal_pow_mean(tensor):
    return torch.mean(tensor**2)
def cal_rms(tensor):
    return np.sqrt(np.mean(tensor**2))
def cal_L1(tensor):
    return torch.mean(abs(tensor))
def data_generate(x, y, z, t):
    #a, d = np.pi/4, np.pi/2
    a, d = 1, 1
    u = - a * (torch.exp(a * x) * torch.sin(a * y + d * z) + torch.exp(a * z) * torch.cos(a * x + d * y)) * torch.exp(- d * d * t)
    v = - a * (torch.exp(a * y) * torch.sin(a * z + d * x) + torch.exp(a * x) * torch.cos(a * y + d * z)) * torch.exp(- d * d * t)
    w = - a * (torch.exp(a * z) * torch.sin(a * x + d * y) + torch.exp(a * y) * torch.cos(a * z + d * x)) * torch.exp(- d * d * t)
    p = - 0.5 * a * a * (torch.exp(2 * a * x) + torch.exp(2 * a * y) + torch.exp(2 * a * z) +
                         2 * torch.sin(a * x + d * y) * torch.cos(a * z + d * x) * torch.exp(a * (y + z)) +
                         2 * torch.sin(a * y + d * z) * torch.cos(a * x + d * y) * torch.exp(a * (z + x)) +
                         2 * torch.sin(a * z + d * x) * torch.cos(a * y + d * z) * torch.exp(a * (x + y))) * torch.exp(
        -2 * d * d * t)
    return u.float(), v.float(), w.float(), p.float()
def taylor_green_flow(x,y,z,t,vis):
    A=2
    B=2
    C=2
    k=2
    miu=vis
    rou=1
    u=(A*torch.sin(k*z)+C*torch.cos(k*y))*torch.exp(-miu*k**2*t)
    v=(B*torch.sin(k*x)+A*torch.cos(k*z))*torch.exp(-miu*k**2*t)
    w=(C*torch.sin(k*y)+C*torch.cos(k*x))*torch.exp(-miu*k**2*t)
    p=-rou*(B*C*torch.cos(k*x)*torch.sin(k*y)+A*B*torch.cos(k*z)*torch.sin(k*x)+
            A*C*torch.cos(k*y)*torch.sin(k*z))*torch.exp(-2*miu*k**2*t)
    return u.float(), v.float(), w.float(), p.float()
def gen_flow(xg, yg, zg, tg):
    u = []
    v = []
    w = []
    p = []
    for t in tg:
        ui, vi, wi, pi = data_generate(xg, yg, zg, t)
        u.append(ui)
        v.append(vi)
        w.append(wi)
        p.append(pi)
    u = torch.stack(u, dim=0)
    v = torch.stack(v, dim=0)
    w = torch.stack(w, dim=0)
    p = torch.stack(p, dim=0)
    return u, v, w, p
def gen_taylor_green_flow(xg, yg, zg, tg,vis):
    u = []
    v = []
    w = []
    p = []
    for t in tg:
        ui, vi, wi, pi = taylor_green_flow(xg, yg, zg, t,vis)
        u.append(ui)
        v.append(vi)
        w.append(wi)
        p.append(pi)
    u = torch.stack(u, dim=0)
    v = torch.stack(v, dim=0)
    w = torch.stack(w, dim=0)
    p = torch.stack(p, dim=0)
    return u, v, w, p
def cal_t_diff_pad(u):
    t_step = u.shape[0]
    u_t = []
    u_t_0 = u[1] - u[0]
    u_t.append(u_t_0)
    for i in range(1, t_step - 1):
        u0 = u[i - 1]
        u1 = u[i]
        u2 = u[i + 1]
        u_t.append((u2 - u0) / 2)
        #u_t.append((u2 - u1))
    u_t_end = u[-1] - u[-2]
    u_t.append(u_t_end)
    u_t = torch.stack(u_t, dim=0)
    return u_t
def cal_t_diff(u):
    t_step = u.shape[0]
    u_t = []
    for i in range(1, t_step - 1):
        u0 = u[i - 1]
        u1 = u[i]
        u2 = u[i + 1]
        dert_t_u=torch.ones_like(u)[0][0].unsqueeze(dim=0).unsqueeze(dim=0)*0.0065*0.45
        dert_t_v=torch.zeros_like(u)[0][0].unsqueeze(dim=0).unsqueeze(dim=0).repeat(1,2,1,1,1)
        dert_t_move=torch.cat([dert_t_u,dert_t_v],dim=1)
        # u0_move=forward_warp3d(u0.unsqueeze(dim=1),dert_t_move)[0]
        # u2_move=forward_warp3d(u2.unsqueeze(dim=1),-dert_t_move)[0]
        #u_t.append((u2_move - u0_move) / 2)
        u_t.append((u2 - u0) / 2)
    u_t = torch.stack(u_t, dim=0)
    return u_t
def cal_x_diff(u):
    t_step = u.shape[-1]
    u_x = []
    for i in range(1, t_step - 1):
        u0 = u[:,:,:,:,i - 1]
        u1 = u[:,:,:,:,i]
        u2 = u[:,:,:,:,i + 1]
        u_x.append((u2 - u0) / 2)
    u_x = torch.stack(u_x, dim=4)
    return u_x
def grad_i(u,kernelx,kernely,kernelz,x_step,y_step,z_step,t_step,pad=0):
    if pad == 1:
        RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
    elif pad == 2:
        RepPad = nn.ReplicationPad3d(padding=(2, 2, 2, 2, 2, 2))
    u_x = F.conv3d(u, kernelx, bias=None) / x_step
    u_y = F.conv3d(u, kernely, bias=None) / y_step
    u_z = F.conv3d(u, kernelz, bias=None) / z_step
    if pad:
        u_x = RepPad(u_x)
        u_y = RepPad(u_y)
        u_z = RepPad(u_z)
        u_t = cal_t_diff_pad(u) / t_step
    else:
        u_t=cal_t_diff(u) / t_step
    return u_x,u_y,u_z,u_t
def grad_i_x(u,kernelx,x_step,pad=0):
    if pad==1:
        RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
    elif pad==2:
        RepPad = nn.ReplicationPad3d(padding=(2, 2, 2, 2, 2, 2))
    u_x = F.conv3d(u, kernelx, bias=None,stride=1) / x_step
    if pad:
        u_x = RepPad(u_x)
    return u_x
def get_grad_manufest(u):
    t,b,d,h,w=u.shape
    gridx_u=[]
    gridy_u = []
    gridz_u = []
    for i in range(w-1):
        gridx_u.append(u[:,:,:,:,i+1].cpu().numpy()-u[:,:,:,:,i-1].cpu().numpy())
    for i in range(h-1):
        gridy_u.append(u[:,:,:,i+1,:].cpu().numpy()-u[:,:,:,i-1,:].cpu().numpy())
    for i in range(d-1):
        gridz_u.append(u[:,:,i+1,:,:].cpu().numpy()-u[:,:,i-1,:,:].cpu().numpy())
    gridx_u=np.stack(gridx_u).transpose(1,2,3,4,0)
    gridy_u = np.array(gridy_u).transpose(1,2,3,0,4)
    gridz_u = np.array(gridz_u).transpose(1,2,0,3,4)
    return gridx_u,gridy_u,gridz_u


def cal_NS_residual_cengliu(u, v, w, p, x_step, y_step, z_step, t_step, density=1, viscosity=1.31e-6,pad=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1,1,1))
    kernelx = torch.tensor([[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]],
                            [[-2., 0., 2.],
                             [-4., 0., 4.],
                             [-2., 0., 2.]],
                            [[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]])
    # kernelx = torch.tensor([[[0., 0., 0.],
    #                          [0., 0., 0.],
    #                          [0., 0., 0.]],
    #                         [[0., 0., 0.],
    #                          [-1., 0.,1.],
    #                          [0., 0., 0.]],
    #                         [[0., 0., 0.],
    #                          [0., 0., 0.],
    #                          [0., 0., 0.]]]) / 2

    kernely = kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    kernelx = kernelx.reshape(1, 1, 3, 3, 3).float().cuda()
    kernely = kernely.reshape(1, 1, 3, 3, 3).float().cuda()
    kernelz = kernelz.reshape(1, 1, 3, 3, 3).float().cuda()
#求一阶偏导
    u_x, u_y, u_z, u_t = grad_i(u, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    v_x, v_y, v_z, v_t = grad_i(v, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    w_x, w_y, w_z, w_t = grad_i(w, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    p_x, p_y, p_z, p_t = grad_i(p, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
#求二阶偏导
    u_xx = F.conv3d(u_x, kernelx, bias=None) / x_step
    u_yy = F.conv3d(u_y, kernely, bias=None) / y_step
    u_zz=  F.conv3d(u_z, kernelz, bias=None) / z_step
    if pad:
        u_xx = RepPad(u_xx)
        u_yy = RepPad(u_yy)
        u_zz = RepPad(u_zz)
    v_xx = F.conv3d(v_x, kernelx, bias=None) / x_step
    v_yy = F.conv3d(v_y, kernely, bias=None) / y_step
    v_zz = F.conv3d(v_z, kernelz, bias=None) / z_step
    if pad:
        v_xx = RepPad(v_xx)
        v_yy = RepPad(v_yy)
        v_zz = RepPad(v_zz)
    w_xx = F.conv3d(w_x, kernelx, bias=None) / x_step
    w_yy = F.conv3d(w_y, kernely, bias=None) / y_step
    w_zz=  F.conv3d(w_z, kernelz, bias=None) / z_step
    if pad:
        w_xx = RepPad(w_xx)
        w_yy = RepPad(w_yy)
        w_zz = RepPad(w_zz)
    else:
        u=  u[1:-1,:, 2:-2,2:-2,2:-2]
        v = v[1:-1, :, 2:-2, 2:-2,2:-2]
        w = w[1:-1, :, 2:-2, 2:-2,2:-2]
        u_t=  u_t[:,:,2:-2,2:-2,2:-2]
        v_t = v_t[:, :, 2:-2, 2:-2,2:-2]
        w_t = w_t[:, :, 2:-2, 2:-2,2:-2]

        u_x=  u_x[1:-1, :, 1:-1, 1:-1, 1:-1]
        u_y = u_y[1:-1, :, 1:-1, 1:-1, 1:-1]
        u_z = u_z[1:-1, :, 1:-1, 1:-1, 1:-1]

        v_x=  v_x[1:-1, :, 1:-1, 1:-1, 1:-1]
        v_y = v_y[1:-1, :, 1:-1, 1:-1, 1:-1]
        v_z = v_z[1:-1, :, 1:-1, 1:-1, 1:-1]

        w_x=  w_x[1:-1, :, 1:-1, 1:-1, 1:-1]
        w_y = w_y[1:-1, :, 1:-1, 1:-1, 1:-1]
        w_z = w_z[1:-1, :, 1:-1, 1:-1, 1:-1]

        p_x=  p_x[1:-1, :, 1:-1, 1:-1, 1:-1]
        p_y = p_y[1:-1, :, 1:-1, 1:-1, 1:-1]
        p_z = p_z[1:-1, :, 1:-1, 1:-1, 1:-1]

        u_xx=  u_xx[1:-1,:,:,:]
        u_yy = u_yy[1:-1, :, :, :]
        u_zz = u_zz[1:-1, :, :, :]

        v_xx=  v_xx[1:-1,:,:,:]
        v_yy = v_yy[1:-1, :, :, :]
        v_zz = v_zz[1:-1, :, :, :]

        w_xx=  w_xx[1:-1,:,:,:]
        w_yy = w_yy[1:-1, :, :, :]
        w_zz = w_zz[1:-1, :, :, :]


    res_cont=u_x+v_y+w_z
    res_nsX = u_t + (u * u_x + v * u_y + w * u_z) + 1 / density * p_x - viscosity * (u_xx + u_yy  +u_zz)
    res_nsY = v_t + (u * v_x + v * v_y + w * v_z) + 1 / density * p_y - viscosity * (v_xx + v_yy + v_zz)
    res_nsZ = w_t + (u * w_x + v * w_y + w * w_z) + 1 / density * p_z - viscosity * (w_xx + w_yy + w_zz)

    pow_cont=cal_pow_mean(res_cont)
    pow_nsX=cal_pow_mean(res_nsX)
    pow_nsY=cal_pow_mean(res_nsY)
    pow_nsZ=cal_pow_mean(res_nsZ)
    rms_cont=cal_rms(res_cont)
    rms_nsX = cal_rms(res_nsX)
    rms_nsY = cal_rms(res_nsY)
    rms_nsZ = cal_rms(res_nsZ)



    return res_cont, res_nsX, res_nsY, res_nsZ
def cal_NS_residual_tuanliu(u, v, w, p, x_step, y_step, z_step, t_step, density=1, viscosity=1.31e-6,pad=1,
                            cengliu=0,kernel5=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1,1))
    #Sobel算子
    if kernel5:
        # kernelx = torch.tensor([[[1., -8., 0., 8.,-1.],
        #                          [2., -16., 0., 16.,-2.],
        #                          [4., -32., 0., 32.,-4.],
        #                          [2., -16., 0., 16.,-2.],
        #                          [1., -8., 0., 8.,-1.]],
        #
        #                         [[2., -16., 0., 16.,-2.],
        #                          [4., -32., 0., 32.,-4.],
        #                          [8., -64., 0., 64.,-8.],
        #                          [4., -32., 0., 32.,-4.],
        #                          [2., -16., 0., 16.,-2.]],
        #
        #                         [[4., -32., 0., 32.,-4.],
        #                          [8., -64., 0., 64.,-8.],
        #                          [16., -128., 0., 128.,-16.],
        #                          [8., -64., 0., 64.,-8.],
        #                          [4., -32., 0., 32.,-4.]],
        #
        #                         [[2., -16., 0., 16.,-2.],
        #                          [4., -32., 0., 32.,-4.],
        #                          [8., -64., 0., 64.,-8.],
        #                          [4., -32., 0., 32.,-4.],
        #                          [2., -16., 0., 16.,-2.]],
        #
        #                         [[1., -8., 0., 8., -1.],
        #                          [2., -16., 0., 16., -2.],
        #                          [4., -32., 0., 32., -4.],
        #                          [2., -16., 0., 16., -2.],
        #                          [1., -8., 0., 8., -1.]],
        #                         ])
        kernelx = torch.tensor([[[0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.]],

                                [[0, 0., 0., 0,0.],
                                 [1, -8., 0., 8,-1.],
                                 [2, -16., 0., 16,-2.],
                                 [1, -8., 0., 8,-1.],
                                 [0, 0., 0., 0,0.]],

                                [[0, 0., 0., 0,0.],
                                 [2, -16., 0., 16,-2.],
                                 [4, -32, 0., 32,-4.],
                                 [2, -16., 0., 16,-2.],
                                 [0, 0., 0., 0,0.]],

                                [[0, 0., 0., 0,0.],
                                 [1, -8., 0., 8,-1.],
                                 [2, -16., 0., 16,-2.],
                                 [1, -8., 0., 8,-1.],
                                 [0, 0., 0., 0,0.]],

                                [[0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.],
                                 [0, 0., 0., 0,0.]],
                                ])
        kernelx=kernelx/1200
    else:
        kernelx = torch.tensor([[[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]],
                                [[-2., 0., 2.],
                                 [-4., 0., 4.],
                                 [-2., 0., 2.]],
                                [[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]])
        kernelx=kernelx/torch.sum(abs(kernelx))
        # kernelx = torch.tensor([[[0., -1., 1.],
        #                          [0., -2., 2.],
        #                          [0., -1., 1.]],
        #                         [[0., -2., 2.],
        #                          [0., -4, 4.],
        #                          [0., -2., 2.]],
        #                         [[0., -1., 1.],
        #                          [0., -2., 2.],
        #                          [0., -1., 1.]]])
        # kernelx = kernelx / torch.sum(abs(kernelx))
    kernelx=  kernelx
    kernely =   kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    if kernel5:
        kernelx = kernelx.reshape(1, 1, 5, 5, 5).cuda().double()
        kernely = kernely.reshape(1, 1, 5, 5, 5).cuda().double()
        kernelz = kernelz.reshape(1, 1, 5, 5, 5).cuda().double()
    else:
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).cuda().double()
        kernely = kernely.reshape(1, 1, 3, 3, 3).cuda().double()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).cuda().double()

    #求一阶偏导

    u_x, u_y, u_z, _ = grad_i(u, kernelx, kernely, kernelz, 1, 1, 1, t_step,pad)
    v_x, v_y, v_z, _ = grad_i(v, kernelx, kernely, kernelz, 1, 1, 1, t_step,pad)
    w_x, w_y, w_z, _ = grad_i(w, kernelx, kernely, kernelz, 1, 1, 1, t_step,pad)
    p_x, p_y, p_z, _ = grad_i(p, kernelx, kernely, kernelz, 1, 1, 1, t_step,pad)

    x_diff_grid=grad_i_x(x_step,kernelx,1,pad=pad)
    y_diff_grid=grad_i_x(y_step,kernely,1,pad=pad)
    z_diff_grid=grad_i_x(z_step,kernelz,1,pad=pad)
    u_x_ma,u_y_ma,u_z_ma=get_grad_manufest(u)
    u_x=u_x/x_diff_grid
    u_y=u_y/y_diff_grid
    u_z=u_z/z_diff_grid


    # plt.imshow(u_x.cpu()[ 0, 0 , 10, :],cmap='jet',vmax=1,vmin=-1)
    # plt.colorbar(shrink=0.5)
    # plt.show()


    v_x=v_x/x_diff_grid
    v_y=v_y/y_diff_grid
    v_z=v_z/z_diff_grid

    w_x=w_x/x_diff_grid
    w_y=w_y/y_diff_grid
    w_z=w_z/z_diff_grid

    p_x = p_x / x_diff_grid
    # p_x_mean=p_x.mean(dim=0)
    # plt.subplot(133)
    # plt.imshow(p_x_mean.cpu()[ 0, :, 10, :],cmap='jet',vmax=1,vmin=-1)
    # plt.colorbar(shrink=0.5)
    #plt.show()
    p_y = p_y / y_diff_grid
    p_z = p_z / z_diff_grid
    u_t= cal_t_diff_pad(u)/t_step
    v_t=cal_t_diff_pad(v)/t_step
    w_t = cal_t_diff_pad(w) / t_step
    #X分量
    if cengliu:
        uu_x=  u *u_x
        uv_y = v *u_y
        uw_z = w *u_z
    else:
        uu_x=  grad_i_x( u *u, kernelx, 1,pad)/x_diff_grid
        uv_y = grad_i_x( v *u, kernely, 1,pad)/y_diff_grid
        uw_z = grad_i_x( w *u, kernelz, 1,pad)/z_diff_grid

    #y分量
    if cengliu :
        vu_x=  u*v_x
        vv_y = v*v_y
        vw_z = w*v_z
    else:
        vu_x=  grad_i_x( u *v, kernelx, 1,pad)/x_diff_grid
        vv_y = grad_i_x( v *v, kernely, 1,pad)/y_diff_grid
        vw_z = grad_i_x( w *v, kernelz, 1,pad)/z_diff_grid


    #Z分量
    if cengliu :
        wu_x=  u*w_x
        wv_y = v*w_y
        ww_z = w*w_z
    else:
        wu_x=  grad_i_x( u *w, kernelx, 1,pad)/x_diff_grid
        wv_y = grad_i_x( v *w, kernely, 1,pad)/y_diff_grid
        ww_z = grad_i_x( w *w, kernelz, 1,pad)/z_diff_grid


    #求二阶偏导   粘性项

    u_xx=grad_i_x(u_x,kernelx,1,pad)/x_diff_grid
    u_yy = grad_i_x(u_y, kernely, 1, pad) / y_diff_grid
    u_zz = grad_i_x(u_z, kernelz, 1, pad) / z_diff_grid

    v_xx=grad_i_x(v_x,kernelx,1,pad)/x_diff_grid
    v_yy = grad_i_x(v_y, kernely, 1, pad) / y_diff_grid
    v_zz = grad_i_x(v_z, kernelz, 1, pad) / z_diff_grid

    w_xx=grad_i_x(w_x,kernelx,1,pad)/x_diff_grid
    w_yy = grad_i_x(w_y, kernely, 1, pad) / y_diff_grid
    w_zz = grad_i_x(w_z, kernelz, 1, pad) / z_diff_grid


    res_cont= u_x +v_y + w_z
    res_nsX = u_t + uu_x + uv_y + uw_z + 1 / density * p_x - viscosity * (u_xx + u_yy  +u_zz)
    res_nsY = v_t + vu_x + vv_y + vw_z + 1 / density * p_y - viscosity * (v_xx + v_yy + v_zz)
    res_nsZ = w_t + wu_x + wv_y + ww_z + 1 / density * p_z - viscosity * (w_xx + w_yy + w_zz)


    a_res_cont=res_cont[0,0,10,5:-5,5:-5]
    a_duiliuX=(uu_x + uv_y + uw_z)[0,0,10,5:-5,5:-5]
    a_p_x_pow=1 / density *p_x[0,0,10,5:-5,5:-5]
    a_u_t_pow=u_t[0,0,10,5:-5,5:-5]
    a_vis=-viscosity * (u_xx + u_yy + u_zz)[0,0,10,5:-5,5:-5]
    a_res=a_duiliuX+a_p_x_pow+a_u_t_pow+a_vis
    a_res_mean=torch.mean(abs(a_res))
    a_left=a_u_t_pow+a_duiliuX
    a_right=-a_p_x_pow+a_vis
    plt.subplot(121)
    plt.imshow(a_left.cpu().numpy(),cmap='jet',vmax=2,vmin=-2)
    plt.title('Du/Dt')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(a_right.cpu().numpy(),cmap='jet',vmax=2,vmin=-2)
    plt.title('-(1/rou)*p_x+vis')
    plt.colorbar()
    plt.show()
    plt.figure(figsize=(15, 5))
    nums, bins, patches = plt.hist(res_nsX.cpu().numpy().flatten(), bins=20, edgecolor='k')
    #plt.xticks(bins, bins)
    for num, bin in zip(nums, bins):
        plt.annotate("%.2f" % num, xy=(bin, num), xytext=(bin + 10, num + 5))
    plt.show()


    max_nsx=res_nsX.max()
    pow_cont=cal_pow_mean(res_cont)
    pow_nsX= cal_pow_mean(res_nsX)
    pow_nsY= cal_pow_mean(res_nsY)
    pow_nsZ= cal_pow_mean(res_nsZ)
    pow_left=cal_pow_mean(a_left)
    print('pow_cont='+str(pow_cont))
    print('pow_nsX=' + str(pow_nsX))
    print('pow_nsX/a_left=' + str(pow_nsX/pow_left))
    print('pow_nsY=' + str(pow_nsY))
    print('pow_nsZ=' + str(pow_nsZ))
    mean=np.mean(res_nsX.cpu().numpy().flatten())
    variance=np.var(res_nsX.cpu().numpy().flatten())
    print('mean and var is '+str(mean)+' and '+str(variance))

    # rms_cont= cal_rms(res_cont)
    # rms_nsX = cal_rms(res_nsX)
    # rms_nsY = cal_rms(res_nsY)
    # rms_nsZ = cal_rms(res_nsZ)
    # L1_cont= cal_L1(res_cont)
    # L1_nsX = cal_L1(res_nsX)
    # L1_nsY = cal_L1(res_nsY)
    # L1_nsZ = cal_L1(res_nsZ)
    L1_ns_mean_total=[]
    L1_ns_var_total=[]
    res_cont=res_cont.cpu().numpy()
    res_nsX=res_nsX.cpu().numpy()
    res_nsY = res_nsY.cpu().numpy()
    res_nsZ = res_nsZ.cpu().numpy()
    for l1 in res_nsX:
        L1_ns_mean_total.append(l1.mean())
        L1_ns_var_total.append(l1.var())
    # plt.subplot(221)
    # plt.imshow(abs(res_cont[0,0,:,10]),cmap='jet',vmax=1)
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(abs(res_nsX[0,0,:,10]),cmap='jet',vmax=1)
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(abs(res_nsY[0,0,:,10]),cmap='jet',vmax=1)
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(abs(res_nsZ[0,0,:,10]),cmap='jet',vmax=1)
    # plt.colorbar()
    # plt.show()
    plt.subplot(221)
    plt.imshow(abs(res_cont[0,0,10,:]),cmap='jet',vmax=2)
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(abs(res_nsX[0,0,10,:]),cmap='jet',vmax=2)
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(abs(res_nsY[0,0,10,:]),cmap='jet',vmax=2)
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(abs(res_nsZ[0,0,10,:]),cmap='jet',vmax=2)
    plt.colorbar()
    plt.show()


    return pow_cont, pow_nsX, pow_nsY, pow_nsZ
def quiver_plot_nosave(flow: np.ndarray, norm: bool = False, show: bool = False, stride=1,scale=60):

    u = flow[0,:]
    v = -flow[1,:]
    h, w = u.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    xp, yp = np.meshgrid(x, y)
    plt.quiver(xp[::stride, ::stride], yp[::stride, ::stride], u[::stride, ::stride], v[::stride, ::stride],
               width=0.001, scale=scale)
    plt.gca().invert_yaxis()
def cal_NS_residual_vv(u, v, w, x_step, y_step, z_step, t_step, density=1, viscosity=1.31e-6,pad=0,cengliu=1,kernel5=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1,1,1))
    #Sobel算子
    # x_step=x_step*100
    # y_step = y_step * 100
    # z_step = z_step * 100
    # t_step = t_step * 100
    # kernelx = torch.tensor([[[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]],
    #                         [[-2.,-2., 4.],
    #                          [-4., -4., 8.],
    #                          [-2., -2., 4.]],
    #                         [[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]]])
    if kernel5:
        kernelx = torch.tensor([[[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [2., -54., 0., 54., -2.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]]
                                ])
        kernelx=kernelx/torch.sum(abs(kernelx))
    else:
        kernelx = torch.tensor([[[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]],
                                [[-2., 0., 2.],
                                 [-4., 0., 4.],
                                 [-2., 0., 2.]],
                                [[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]])
        # kernelx = torch.tensor([[[-0., 0., 0.],
        #                          [-0., 0., 0.],
        #                          [-0., 0., 0.]],
        #                         [[-0., 0., 0.],
        #                          [-1., 0., 1.],
        #                          [-0., 0., 0.]],
        #                         [[-0., 0., 0.],
        #                          [-0., 0., 0.],
        #                          [-0., 0., 0.]]])
        kernelx=kernelx/torch.sum(abs(kernelx)).float()
    kernely = kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    if kernel5:
        kernelx = kernelx.reshape(1, 1, 5, 5, 5).float().cuda()
        kernely = kernely.reshape(1, 1, 5, 5, 5).float().cuda()
        kernelz = kernelz.reshape(1, 1, 5, 5, 5).float().cuda()
    else:
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).float().cuda()
        kernely = kernely.reshape(1, 1, 3, 3, 3).float().cuda()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).float().cuda()
    #求一阶偏导
    u=u.float()
    v=v.float()
    w=w.float()

    x_diff_grid=grad_i_x(x_step.float(),kernelx,1,pad=1)
    y_diff_grid=grad_i_x(y_step.float(),kernely,1,pad=1)
    z_diff_grid=grad_i_x(z_step.float(),kernelz,1,pad=1)

    u_x=  grad_i_x(u,kernelx,1,pad=1)
    u_y=  grad_i_x(u,kernely,1,pad=1)
    u_z = grad_i_x(u,kernelz, 1,pad=1)

    v_x=  grad_i_x(v,kernelx,1,pad=1)
    v_y=  grad_i_x(v,kernely,1,pad=1)
    v_z = grad_i_x(v, kernelz, 1,pad=1)

    w_x=  grad_i_x(w,kernelx,1,pad=1)
    w_y=  grad_i_x(w,kernely,1,pad=1)
    w_z = grad_i_x(w, kernelz, 1,pad=1)

    cont = u_x + v_y + w_z
    cont_mean = cal_L1(cont)
    u_x=u_x/x_diff_grid
    u_y=u_y/y_diff_grid
    u_z=u_z/z_diff_grid

    v_x=v_x/x_diff_grid
    v_y=v_y/y_diff_grid
    v_z=v_z/z_diff_grid

    w_x=w_x/x_diff_grid
    w_y=w_y/y_diff_grid
    w_z=w_z/z_diff_grid

    vor_x=0.5*(w_y-v_z)
    vor_y=0.5*(u_z-w_x)
    vor_z=0.5*(v_x-u_y)
    # plt.subplot(311)
    # plt.imshow(vor_x[0,0,:,15,:].cpu(),cmap='jet')
    # plt.subplot(312)
    #quiver_plot_nosave(np.stack([v[5,0,15,:,:].cpu().numpy(), w[5,0,15,:,:].cpu().numpy()], axis=0))
    #plt.imshow(vor_z[5,0,15,:,:].cpu(),cmap='jet',vmax=0.5,vmin=-0.5,alpha=0.7)

    #plt.axis('off')
    #plt.imshow(vorz[10, 0, 15, :, :].cpu(), cmap='jet')
    #plt.colorbar()
    # plt.subplot(313)
    # plt.imshow(vor_z[0,0,:,15,:].cpu(),cmap='jet')
    #plt.show()

    vor_x_rms=  cal_rms(vor_x)
    vor_y_rms = cal_rms(vor_y)
    vor_z_rms = cal_rms(vor_z)
    evv_vor_z_t=cal_t_diff_pad(vor_z)/t_step
    evv1_1=(u*vor_z-w*vor_x)
    evv1_1=grad_i_x(evv1_1,kernelx,1,pad=1)/x_diff_grid
    evv1_2=(w*vor_y-v*vor_z)
    evv1_2=grad_i_x(evv1_2,kernely,1,pad=1)/y_diff_grid
    evv1_3=grad_i_x((grad_i_x(vor_x,kernelz,1,pad=1)/z_diff_grid
                     -grad_i_x(vor_z,kernelx,1,pad=1)/x_diff_grid),
                    kernelx,1,pad=1)/x_diff_grid
    evv1_4 = grad_i_x((grad_i_x(vor_z, kernely, 1,pad=1) / y_diff_grid -
                       grad_i_x(vor_y, kernelz, 1,pad=1) / z_diff_grid)
                    , kernely,1,pad=1) / y_diff_grid
    evv1_3_4=viscosity*(evv1_3-evv1_4)
    u_x=u_x.cpu().numpy()[:, :, 1:-1, 1:-1, 1:-1]
    v_y=v_y.cpu().numpy()[:, :, 1:-1, 1:-1, 1:-1]
    w_z=w_z.cpu().numpy()[:, :, 1:-1, 1:-1, 1:-1]
    cont=u_x+v_y+w_z
    cont=cont

    evv_vor_z_t=evv_vor_z_t
    evv1_1= evv1_1
    evv1_2=-evv1_2
    evv1=evv_vor_z_t+evv1_1+evv1_2+evv1_3_4
    evv1_L1=cal_L1(evv1)
    evv1_mean=evv1.mean()
    return evv1_L1
def cal_NS_residual_vv_rotate(u, v, w, x_step, y_step, z_step, t_step, density=1, viscosity=1.31e-6,cengliu=1,kernel5=0):

    #Sobel算子
    # x_step=x_step*100
    # y_step = y_step * 100
    # z_step = z_step * 100
    # t_step = t_step * 100
    # kernelx = torch.tensor([[[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]],
    #                         [[-2.,-2., 4.],
    #                          [-4., -4., 8.],
    #                          [-2., -2., 4.]],
    #                         [[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]]])
    if kernel5:
        kernelx = torch.tensor([[[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -8., 0., 8., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [1., -8., 0., 8., -1.],
                                 [2., -16., 0., 16., -2.],
                                 [1., -8., 0., 8., -1.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -8., 0., 8., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],
                                ])
        kernelx=kernelx/12/6
    else:
        # kernelx = torch.tensor([[[-1., 0., 1.],
        #                          [-2., 0., 2.],
        #                          [-1., 0., 1.]],
        #                         [[-2., 0., 2.],
        #                          [-4., 0., 4.],
        #                          [-2., 0., 2.]],
        #                         [[-1., 0., 1.],
        #                          [-2., 0., 2.],
        #                          [-1., 0., 1.]]])
        kernelx = torch.tensor([[[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-1., 0., 1.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]]])
        kernelx=kernelx/torch.sum(abs(kernelx)).float()
    kernely =kernelx.transpose(1, 2)
    kernelz =kernelx.transpose(2, 0)
    if kernel5:
        kernelx = kernelx.reshape(1, 1, 5, 5, 5).float().cuda()
        kernely = kernely.reshape(1, 1, 5, 5, 5).float().cuda()
        kernelz = kernelz.reshape(1, 1, 5, 5, 5).float().cuda()
    else:
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).float().cuda()
        kernely = kernely.reshape(1, 1, 3, 3, 3).float().cuda()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).float().cuda()
    if kernel5:
        pad_num=2
    else:
        pad_num=1
    #求一阶偏导
    u=u.float()
    v=v.float()
    w=w.float()

    x_diff_grid=abs(grad_i_x(x_step.float(),kernelx,1,pad=pad_num))
    y_diff_grid=abs(grad_i_x(y_step.float(),kernely,1,pad=pad_num))
    z_diff_grid=abs(grad_i_x(z_step.float(),kernelz,1,pad=pad_num))

    u_x=  grad_i_x(u,kernelx, 1,pad=pad_num)/x_diff_grid
    u_y=  grad_i_x(u,kernely, 1,pad=pad_num)/y_diff_grid
    u_z = grad_i_x(u,kernelz, 1,pad=pad_num)/z_diff_grid

    v_x=  grad_i_x(v, kernelx,1,pad=pad_num)/x_diff_grid
    v_y=  grad_i_x(v, kernely,1,pad=pad_num)/y_diff_grid
    v_z = grad_i_x(v, kernelz, 1,pad=pad_num)/z_diff_grid

    w_x=  grad_i_x(w,kernelx,1,pad=pad_num)/x_diff_grid
    w_y=  grad_i_x(w,kernely,1,pad=pad_num)/y_diff_grid
    w_z = grad_i_x(w, kernelz, 1,pad=pad_num)/z_diff_grid
    cont=u_x+v_y+w_z

    oumiga_x=(w_y-v_z)
    oumiga_y=(u_z-w_x)
    oumiga_z=(v_x-u_y)

    oumiga_x_t = (cal_t_diff_pad(oumiga_x) / t_step)
    oumiga_y_t = (cal_t_diff_pad(oumiga_y) / t_step)
    oumiga_z_t = (cal_t_diff_pad(oumiga_z) / t_step)
    a=v*oumiga_x-u*oumiga_y
    b=w*oumiga_y-v*oumiga_z
    c=u*oumiga_z-w*oumiga_x
    oumiga_x_x = (grad_i_x(oumiga_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_x_x2 = (grad_i_x(oumiga_x_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_x_y=(grad_i_x(oumiga_x, kernely, 1, pad=pad_num)/y_diff_grid)
    oumiga_x_y2 = (grad_i_x(oumiga_x_y, kernely, 1, pad=pad_num) /y_diff_grid)
    oumiga_x_z=(grad_i_x(oumiga_x, kernelz, 1, pad=pad_num)/z_diff_grid)
    oumiga_x_z2 = (grad_i_x(oumiga_x_z, kernelz, 1, pad=pad_num) / z_diff_grid)

    oumiga_y_x = (grad_i_x(oumiga_y, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_y_x2 = (grad_i_x(oumiga_y_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_y_y=(grad_i_x(oumiga_y, kernely, 1, pad=pad_num)/y_diff_grid)
    oumiga_y_y2 = (grad_i_x(oumiga_y_y, kernely, 1, pad=pad_num) /y_diff_grid)
    oumiga_y_z=(grad_i_x(oumiga_y, kernelz, 1, pad=pad_num)/z_diff_grid)
    oumiga_y_z2 = (grad_i_x(oumiga_y_z, kernelz, 1, pad=pad_num) / z_diff_grid)

    oumiga_z_x = (grad_i_x(oumiga_z, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_z_x2 = (grad_i_x(oumiga_z_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_z_y=(grad_i_x(oumiga_z, kernely, 1, pad=pad_num)/y_diff_grid)
    oumiga_z_y2 = (grad_i_x(oumiga_z_y, kernely, 1, pad=pad_num) /y_diff_grid)
    oumiga_z_z=(grad_i_x(oumiga_z, kernelz, 1, pad=pad_num)/z_diff_grid)
    oumiga_z_z2 = (grad_i_x(oumiga_z_z, kernelz, 1, pad=pad_num) / z_diff_grid)

    a_x = (grad_i_x(a, kernelx, x_step=1, pad=pad_num) / x_diff_grid)
    a_y=  (grad_i_x(a, kernely,x_step=1,pad=pad_num)/y_diff_grid)
    a_z = (grad_i_x(a, kernelz, x_step=1, pad=pad_num) / z_diff_grid)

    b_x = (grad_i_x(b, kernelx, x_step=1, pad=pad_num) / x_diff_grid)
    b_y=(grad_i_x(b, kernely,x_step=1,pad=pad_num)/y_diff_grid)
    b_z = (grad_i_x(b, kernelz, x_step=1, pad=pad_num) / z_diff_grid)

    c_x = (grad_i_x(c, kernelx, x_step=1, pad=pad_num) / x_diff_grid)
    c_y = (grad_i_x(c, kernely,x_step=1,pad=pad_num)/y_diff_grid)
    c_z = (grad_i_x(c, kernelz, x_step=1, pad=pad_num) / z_diff_grid)


    aresx=-oumiga_x_t-a_y+c_z+viscosity*(oumiga_x_x2+oumiga_x_y2+oumiga_x_z2)
    aresy=-oumiga_y_t+a_x-b_z+viscosity*(oumiga_y_x2+oumiga_y_y2+oumiga_y_z2)
    aresz = -oumiga_z_t - c_x + b_y + viscosity * (oumiga_z_x2 + oumiga_z_y2 + oumiga_z_z2)

    aresx = aresx[1:-1,:,2:-2,2:-2,2:-2]/cal_rms(oumiga_x_t[1:-1,:,2:-2,2:-2,2:-2].detach().cpu().numpy())
    aresy = aresy[1:-1,:,2:-2,2:-2,2:-2]/cal_rms(oumiga_y_t[1:-1,:,2:-2,2:-2,2:-2].detach().cpu().numpy())
    aresz = aresz[1:-1,:,2:-2,2:-2,2:-2]/cal_rms(oumiga_z_t[1:-1,:,2:-2,2:-2,2:-2].detach().cpu().numpy())

    u_np = u.detach().cpu().numpy()

    aresx_np=  aresx.detach().cpu().numpy()
    aresy_np = aresy.detach().cpu().numpy()
    aresz_np = aresz.detach().cpu().numpy()






    aresx_np[aresx_np>10]=0
    aresy_np[aresy_np > 10] = 0
    aresz_np[aresz_np > 10] = 0

    aresx_np[aresx_np<-10]=0
    aresy_np[aresy_np < -10] = 0
    aresz_np[aresz_np < -10] = 0

    # plt.subplot(331)
    # plt.imshow(u_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(332)
    # plt.imshow(v_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(333)
    # plt.imshow(w_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(334)
    # plt.imshow(ox_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(335)
    # plt.imshow(oy_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(336)
    # plt.imshow(oz_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(337)
    # plt.imshow(aresx_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(338)
    # plt.imshow(aresy_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(339)
    # plt.imshow(aresz_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.show()
    #
    # ares_mean = np.mean(aresx_np.flatten())
    # ares_var = np.std(aresx_np.flatten())
    # print('mean and var=' + str(ares_mean) + ' and ' + str(ares_var))
    # nums, bins, patches = plt.hist(aresx_np.flatten(), bins=20, edgecolor='k')
    # # plt.xticks(bins, bins)
    # for num, bin in zip(nums, bins):
    #     plt.annotate("%.2f" % num, xy=(bin, num), xytext=(bin + 10, num + 5))
    # plt.show()



    return aresx,aresy,aresz, cont/((abs(u_x)+abs(v_y)+abs(w_z))/3)
def cal_NS_residual_vv_rotate_nonorm(u, v, w, x_step, y_step, z_step, t_step, density=1, viscosity=1.31e-6,cengliu=1,kernel5=0):

    #Sobel算子
    # x_step=x_step*100
    # y_step = y_step * 100
    # z_step = z_step * 100
    # t_step = t_step * 100
    # kernelx = torch.tensor([[[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]],
    #                         [[-2.,-2., 4.],
    #                          [-4., -4., 8.],
    #                          [-2., -2., 4.]],
    #                         [[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]]])
    if kernel5:
        kernelx = torch.tensor([[[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -8., 0., 8., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [1., -8., 0., 8., -1.],
                                 [2., -16., 0., 16., -2.],
                                 [1., -8., 0., 8., -1.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -8., 0., 8., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],
                                ])
        kernelx=kernelx/12/6
    else:
        # kernelx = torch.tensor([[[-1., 0., 1.],
        #                          [-2., 0., 2.],
        #                          [-1., 0., 1.]],
        #                         [[-2., 0., 2.],
        #                          [-4., 0., 4.],
        #                          [-2., 0., 2.]],
        #                         [[-1., 0., 1.],
        #                          [-2., 0., 2.],
        #                          [-1., 0., 1.]]])
        kernelx = torch.tensor([[[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-1., 0., 1.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]]])
        kernelx=kernelx/torch.sum(abs(kernelx)).float()
    kernely =kernelx.transpose(1, 2)
    kernelz =kernelx.transpose(2, 0)
    if kernel5:
        kernelx = kernelx.reshape(1, 1, 5, 5, 5).float().cuda()
        kernely = kernely.reshape(1, 1, 5, 5, 5).float().cuda()
        kernelz = kernelz.reshape(1, 1, 5, 5, 5).float().cuda()
    else:
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).float().cuda()
        kernely = kernely.reshape(1, 1, 3, 3, 3).float().cuda()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).float().cuda()
    if kernel5:
        pad_num=2
    else:
        pad_num=1
    #求一阶偏导
    u=u.float()
    v=v.float()
    w=w.float()

    x_diff_grid=abs(grad_i_x(x_step.float(),kernelx,1,pad=pad_num))
    y_diff_grid=abs(grad_i_x(y_step.float(),kernely,1,pad=pad_num))
    z_diff_grid=abs(grad_i_x(z_step.float(),kernelz,1,pad=pad_num))

    u_x=  grad_i_x(u,kernelx, 1,pad=pad_num)/x_diff_grid
    u_y=  grad_i_x(u,kernely, 1,pad=pad_num)/y_diff_grid
    u_z = grad_i_x(u,kernelz, 1,pad=pad_num)/z_diff_grid

    v_x=  grad_i_x(v, kernelx,1,pad=pad_num)/x_diff_grid
    v_y=  grad_i_x(v, kernely,1,pad=pad_num)/y_diff_grid
    v_z = grad_i_x(v, kernelz, 1,pad=pad_num)/z_diff_grid

    w_x=  grad_i_x(w,kernelx,1,pad=pad_num)/x_diff_grid
    w_y=  grad_i_x(w,kernely,1,pad=pad_num)/y_diff_grid
    w_z = grad_i_x(w, kernelz, 1,pad=pad_num)/z_diff_grid
    cont=u_x+v_y+w_z

    oumiga_x=(w_y-v_z)
    oumiga_y=(u_z-w_x)
    oumiga_z=(v_x-u_y)

    oumiga_x_t = (cal_t_diff_pad(oumiga_x) / t_step)
    oumiga_y_t = (cal_t_diff_pad(oumiga_y) / t_step)
    oumiga_z_t = (cal_t_diff_pad(oumiga_z) / t_step)
    a=v*oumiga_x-u*oumiga_y
    b=w*oumiga_y-v*oumiga_z
    c=u*oumiga_z-w*oumiga_x
    oumiga_x_x = (grad_i_x(oumiga_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_x_x2 = (grad_i_x(oumiga_x_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_x_y=(grad_i_x(oumiga_x, kernely, 1, pad=pad_num)/y_diff_grid)
    oumiga_x_y2 = (grad_i_x(oumiga_x_y, kernely, 1, pad=pad_num) /y_diff_grid)
    oumiga_x_z=(grad_i_x(oumiga_x, kernelz, 1, pad=pad_num)/z_diff_grid)
    oumiga_x_z2 = (grad_i_x(oumiga_x_z, kernelz, 1, pad=pad_num) / z_diff_grid)

    oumiga_y_x = (grad_i_x(oumiga_y, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_y_x2 = (grad_i_x(oumiga_y_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_y_y=(grad_i_x(oumiga_y, kernely, 1, pad=pad_num)/y_diff_grid)
    oumiga_y_y2 = (grad_i_x(oumiga_y_y, kernely, 1, pad=pad_num) /y_diff_grid)
    oumiga_y_z=(grad_i_x(oumiga_y, kernelz, 1, pad=pad_num)/z_diff_grid)
    oumiga_y_z2 = (grad_i_x(oumiga_y_z, kernelz, 1, pad=pad_num) / z_diff_grid)

    oumiga_z_x = (grad_i_x(oumiga_z, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_z_x2 = (grad_i_x(oumiga_z_x, kernelx, 1, pad=pad_num)/x_diff_grid)
    oumiga_z_y=(grad_i_x(oumiga_z, kernely, 1, pad=pad_num)/y_diff_grid)
    oumiga_z_y2 = (grad_i_x(oumiga_z_y, kernely, 1, pad=pad_num) /y_diff_grid)
    oumiga_z_z=(grad_i_x(oumiga_z, kernelz, 1, pad=pad_num)/z_diff_grid)
    oumiga_z_z2 = (grad_i_x(oumiga_z_z, kernelz, 1, pad=pad_num) / z_diff_grid)

    a_x = (grad_i_x(a, kernelx, x_step=1, pad=pad_num) / x_diff_grid)
    a_y=  (grad_i_x(a, kernely,x_step=1,pad=pad_num)/y_diff_grid)
    a_z = (grad_i_x(a, kernelz, x_step=1, pad=pad_num) / z_diff_grid)

    b_x = (grad_i_x(b, kernelx, x_step=1, pad=pad_num) / x_diff_grid)
    b_y=(grad_i_x(b, kernely,x_step=1,pad=pad_num)/y_diff_grid)
    b_z = (grad_i_x(b, kernelz, x_step=1, pad=pad_num) / z_diff_grid)

    c_x = (grad_i_x(c, kernelx, x_step=1, pad=pad_num) / x_diff_grid)
    c_y = (grad_i_x(c, kernely,x_step=1,pad=pad_num)/y_diff_grid)
    c_z = (grad_i_x(c, kernelz, x_step=1, pad=pad_num) / z_diff_grid)


    aresx=-oumiga_x_t-a_y+c_z+viscosity*(oumiga_x_x2+oumiga_x_y2+oumiga_x_z2)
    aresy=-oumiga_y_t+a_x-b_z+viscosity*(oumiga_y_x2+oumiga_y_y2+oumiga_y_z2)
    aresz = -oumiga_z_t - c_x + b_y + viscosity * (oumiga_z_x2 + oumiga_z_y2 + oumiga_z_z2)

    aresx = aresx[1:-1,:,2:-2,2:-2,2:-2]
    aresy = aresy[1:-1,:,2:-2,2:-2,2:-2]
    aresz = aresz[1:-1,:,2:-2,2:-2,2:-2]
    cont=cont[1:-1,:,1:-1,1:-1,1:-1]
    # plt.subplot(331)
    # plt.imshow(u_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(332)
    # plt.imshow(v_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(333)
    # plt.imshow(w_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(334)
    # plt.imshow(ox_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(335)
    # plt.imshow(oy_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(336)
    # plt.imshow(oz_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(337)
    # plt.imshow(aresx_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(338)
    # plt.imshow(aresy_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.subplot(339)
    # plt.imshow(aresz_np[2,0,10,:,:],cmap='jet')
    # plt.colorbar(shrink=0.5)
    # plt.show()
    #
    # ares_mean = np.mean(aresx_np.flatten())
    # ares_var = np.std(aresx_np.flatten())
    # print('mean and var=' + str(ares_mean) + ' and ' + str(ares_var))
    # nums, bins, patches = plt.hist(aresx_np.flatten(), bins=20, edgecolor='k')
    # # plt.xticks(bins, bins)
    # for num, bin in zip(nums, bins):
    #     plt.annotate("%.2f" % num, xy=(bin, num), xytext=(bin + 10, num + 5))
    # plt.show()



    return aresx,aresy,aresz, cont

def cal_NS_residual_vv_numpy(u,v,w1):
    # 读取速度数据
    # 定义网格大小和物理参数
    u=u[:,0].cpu().numpy()
    v = v[:, 0].cpu().numpy()
    w1 = w1[:, 0].cpu().numpy()
    t, d, h, w = 10, 32, 32, 32
    # 随机生成速度分量
    # 定义涡量
    omega = np.zeros((t, d, h, w, 3))
    # 计算涡量

    for k in range(1, w - 1):
        for j in range(1, d - 1):
            for i in range(1, h - 1):
                omega[:, j, i, k, 0] = w1[:, j + 1, i, k] - w1[:, j - 1, i, k] - v[:, j, i + 1, k] + v[:, j, i - 1, k]
                omega[:, j, i, k, 1] = u[:, j, i + 1, k] - u[:, j, i - 1, k] - w1[:, j, i, k + 1] + w1[:, j, i, k - 1]
                omega[:, j, i, k, 2] = v[:, j + 1, i, k] - v[:, j - 1, i, k] - u[:, j, i, k + 1] + u[:, j, i, k - 1]
    # 定义残差数组
    residual = np.zeros((t, d, h, w, 3))
    # 定义时间步长和运动粘性系数
    dt = 0.01
    nu = 1
    step=0.00625
    # 计算涡量输送方程的残差
    for k in range(1, w - 1):
        for j in range(1, d - 1):
            for i in range(1, h - 1):
                residual[:, j, i, k, 0] = (omega[:, j, i, k, 0] - omega[:, j, i, k - 1, 0]) / dt \
                                          + u[:, j, i, k] * (omega[:, j, i + 1, k, 0] - omega[:, j, i - 1, k, 0]) / (
                                                      2 * step) \
                                          + v[:, j, i, k] * (omega[:, j + 1, i, k, 0] - omega[:, j - 1, i, k, 0]) / (
                                                      2 * step) \
                                          + w1[:, j, i, k] * (omega[:, j, i, k + 1, 0] - omega[:, j, i, k - 1, 0]) / (
                                                      2 * step) \
                                          + nu * ((omega[:, j, i + 1, k, 0] - 2 * omega[:, j, i, k, 0] + omega[:, j,
                                                                                                         i - 1, k,
                                                                                                         0]) / (step ** 2) \
                                                  + (omega[:, j + 1, i, k, 0] - 2 * omega[:, j, i, k, 0] + omega[:,
                                                                                                           j - 1, i, k,
                                                                                                           0]) / (
                                                              step ** 2) \
                                                  + (omega[:, j, i, k + 1, 0] - 2 * omega[:, j, i, k, 0] + omega[:, j,
                                                                                                           i, k - 1,
                                                                                                           0]) / (
                                                              step ** 2))
                residual[:, j, i, k, 1] = (omega[:, j, i, k, 1] - omega[:, j, i, k - 1, 1]) / dt \
                                          + u[:, j, i, k] * (omega[:, j, i + 1, k, 1] - omega[:, j, i - 1, k, 1]) / (
                                                      2 * step) \
                                          + v[:, j, i, k] * (omega[:, j + 1, i, k, 1] - omega[:, j - 1, i, k, 1]) / (
                                                      2 * step) \
                                          + w1[:, j, i, k] * (omega[:, j, i, k + 1, 1] - omega[:, j, i, k - 1, 1]) / (
                                                      2 * step) \
                                          + nu * ((omega[:, j, i + 1, k, 1] - 2 * omega[:, j, i, k, 1] + omega[:, j,
                                                                                                         i - 1, k,
                                                                                                         1]) / (step ** 2) \
                                                  + (omega[:, j + 1, i, k, 1] - 2 * omega[:, j, i, k, 1] + omega[:,
                                                                                                           j - 1, i, k,
                                                                                                           1]) / (
                                                              step ** 2) \
                                                  + (omega[:, j, i, k + 1, 1] - 2 * omega[:, j, i, k, 1] + omega[:, j,
                                                                                                           i, k - 1,
                                                                                                           1]) / (
                                                              step ** 2))
                residual[:, j, i, k, 2] = (omega[:, j, i, k, 2] - omega[:, j, i, k - 1, 2]) / dt \
                                          + u[:, j, i, k] * (omega[:, j, i + 1, k, 2] - omega[:, j, i - 1, k, 2]) / (
                                                      2 * step) \
                                          + v[:, j, i, k] * (omega[:, j + 1, i, k, 2] - omega[:, j - 1, i, k, 2]) / (
                                                      2 * step) \
                                          + w1[:, j, i, k] * (omega[:, j, i, k + 1, 2] - omega[:, j, i, k - 1, 2]) / (
                                                      2 * step) \
                                          + nu * ((omega[:, j, i + 1, k, 2] - 2 * omega[:, j, i, k, 2] + omega[:, j,
                                                                                                         i - 1, k,
                                                                                                         2]) / (step ** 2) \
                                                  + (omega[:, j + 1, i, k, 2] - 2 * omega[:, j, i, k, 2] + omega[:,
                                                                                                           j - 1, i, k,
                                                                                                           2]) / (
                                                              step ** 2) \
                                                  + (omega[:, j, i, k + 1, 2] - 2 * omega[:, j, i, k, 2] + omega[:, j,
                                                                                                           i, k - 1,
                                                                                                           2]) / (

                                                              step ** 2))
    # 输出残差数组的形状
    residual=residual.transpose(4,0,1,2,3)[:,:,2:-2,2:-2,2:-2]
    mean_residual=np.mean(abs(residual))
    print(residual.shape)
def cal_NS_residual_tuanliu_Re(u, v, w, p, x_step, y_step, z_step, t_step,Re,pad=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1,1,1))
    kernelx = torch.tensor([[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]],
                            [[-2., 0., 2.],
                             [-4., 0., 4.],
                             [-2., 0., 2.]],
                            [[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]]) / 32
    kernely = kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    kernelx = kernelx.reshape(1, 1, 3, 3, 3).float().cuda()
    kernely = kernely.reshape(1, 1, 3, 3, 3).float().cuda()
    kernelz = kernelz.reshape(1, 1, 3, 3, 3).float().cuda()
#求一阶偏导
    u_x, u_y, u_z, u_t = grad_i(u, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    v_x, v_y, v_z, v_t = grad_i(v, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    w_x, w_y, w_z, w_t = grad_i(w, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    p_x, p_y, p_z, p_t = grad_i(p, kernelx, kernely, kernelz, x_step, y_step, z_step, t_step,pad)
    #X分量
    uu_x=  grad_i_x(u * u, kernelx, x_step)
    uv_y = grad_i_x(v * u, kernely, y_step)
    uw_z = grad_i_x(w * u, kernelz, x_step)
    u_t=cal_t_diff(u)/t_step
    #y分量
    vu_x=  grad_i_x(v*u,kernelx,x_step)
    vv_y = grad_i_x(v * v, kernely, y_step)
    vw_z = grad_i_x(w * v, kernelz, x_step)
    v_t=cal_t_diff(v)/t_step
    #Z分量
    wu_x=  grad_i_x(w*u,kernelx,x_step)
    wv_y = grad_i_x(v * w, kernely, y_step)
    ww_z = grad_i_x(w * w, kernelz, x_step)
    w_t=cal_t_diff(w)/t_step

#求二阶偏导
    u_xx = F.conv3d(u_x, kernelx, bias=None) / x_step
    u_yy = F.conv3d(u_y, kernely, bias=None) / y_step
    u_zz=  F.conv3d(u_z, kernelz, bias=None) / z_step
    if pad:
        u_xx = RepPad(u_xx)
        u_yy = RepPad(u_yy)
        u_zz = RepPad(u_zz)
    v_xx = F.conv3d(v_x, kernelx, bias=None) / x_step
    v_yy = F.conv3d(v_y, kernely, bias=None) / y_step
    v_zz = F.conv3d(v_z, kernelz, bias=None) / z_step
    if pad:
        v_xx = RepPad(v_xx)
        v_yy = RepPad(v_yy)
        v_zz = RepPad(v_zz)
    w_xx = F.conv3d(w_x, kernelx, bias=None) / x_step
    w_yy = F.conv3d(w_y, kernely, bias=None) / y_step
    w_zz=  F.conv3d(w_z, kernelz, bias=None) / z_step

    u=  u[1:-1,:,2:-2,2:-2,2:-2]
    v = v[1:-1, :, 2:-2, 2:-2,2:-2]
    w = w[1:-1, :, 2:-2, 2:-2,2:-2]
    u_t=  u_t[:,:,2:-2,2:-2,2:-2]
    v_t = v_t[:, :, 2:-2, 2:-2,2:-2]
    w_t = w_t[:, :, 2:-2, 2:-2,2:-2]

    uu_x=  uu_x[1:-1, :, 1:-1, 1:-1, 1:-1]
    uv_y = uv_y[1:-1, :, 1:-1, 1:-1, 1:-1]
    uw_z = uw_z[1:-1, :, 1:-1, 1:-1, 1:-1]

    vu_x=  vu_x[1:-1, :, 1:-1, 1:-1, 1:-1]
    vv_y = vv_y[1:-1, :, 1:-1, 1:-1, 1:-1]
    vw_z = vw_z[1:-1, :, 1:-1, 1:-1, 1:-1]

    wu_x=  wu_x[1:-1, :, 1:-1, 1:-1, 1:-1]
    wv_y = wv_y[1:-1, :, 1:-1, 1:-1, 1:-1]
    ww_z = ww_z[1:-1, :, 1:-1, 1:-1, 1:-1]

    p_x=p_x[1:-1, :, 1:-1, 1:-1, 1:-1]
    p_y = p_y[1:-1, :, 1:-1, 1:-1, 1:-1]
    p_z = p_z[1:-1, :, 1:-1, 1:-1, 1:-1]
    u_xx=  u_xx[1:-1,:,:,:]
    u_yy = u_yy[1:-1, :, :, :]
    u_zz = u_zz[1:-1, :, :, :]

    v_xx=  v_xx[1:-1,:,:,:]
    v_yy = v_yy[1:-1, :, :, :]
    v_zz = v_zz[1:-1, :, :, :]

    w_xx=  w_xx[1:-1,:,:,:]
    w_yy = w_yy[1:-1, :, :, :]
    w_zz = w_zz[1:-1, :, :, :]
    res_cont=u_x+v_y+w_z
    res_nsX = u_t + (uu_x + uv_y + uw_z) +  p_x - 1/Re * (u_xx + u_yy  +u_zz)
    res_nsY = v_t + (vu_x + vv_y + vw_z) +  p_y - 1/Re * (v_xx + v_yy + v_zz)
    res_nsZ = w_t + (wu_x + wv_y + ww_z) +  p_z - 1/Re * (w_xx + w_yy + w_zz)
    pow_cont=cal_pow_mean(res_cont)
    pow_nsX=cal_pow_mean(res_nsX)
    pow_nsY=cal_pow_mean(res_nsY)
    pow_nsZ=cal_pow_mean(res_nsZ)
    return pow_cont, pow_nsX, pow_nsY, pow_nsZ
def cal_P_by_NS(u, v, w,x_step, y_step, z_step, t_step,density=1, viscosity=5e-5,pad=0,kernel5=0,cengliu=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
    # Sobel算子

    # kernelx = torch.tensor([[[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]],
    #                         [[-2.,-2., 4.],
    #                          [-4., -4., 8.],
    #                          [-2., -2., 4.]],
    #                         [[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]]])
    if kernel5:
        kernelx = torch.tensor([[[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [2., -54., 0., 54., -2.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]]
                                ])
        kernelx = kernelx / torch.sum(abs(kernelx))
    else:
        kernelx = torch.tensor([[[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-1., 0., 1.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]]])

        kernelx = kernelx / torch.sum(abs(kernelx))
    # kernelx = torch.tensor([[[0., 0., 0.],
    #                          [0., 0., 0.],
    #                          [0., 0., 0.]],
    #                         [[0., 0., 0.],
    #                          [-1., 0.,1.],
    #                          [0., 0., 0.]],
    #                         [[0., 0., 0.],
    #                          [0., 0., 0.],
    #                          [0., 0., 0.]]]) / 2
    kernely = kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    if kernel5:
        kernelx = kernelx.reshape(1, 1, 5, 5, 5).float().cuda()
        kernely = kernely.reshape(1, 1, 5, 5, 5).float().cuda()
        kernelz = kernelz.reshape(1, 1, 5, 5, 5).float().cuda()
    else:
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).cuda().double()
        kernely = kernely.reshape(1, 1, 3, 3, 3).cuda().double()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).cuda().double()
    # 求一阶偏导
    u_x, u_y, u_z, _ = grad_i(u[1:-1, :, :, :, :], kernelx, kernely, kernelz, 1, 1, 1, t_step, pad)
    v_x, v_y, v_z, _ = grad_i(v[1:-1, :, :, :, :], kernelx, kernely, kernelz, 1, 1, 1, t_step, pad)
    w_x, w_y, w_z, _ = grad_i(w[1:-1, :, :, :, :], kernelx, kernely, kernelz, 1, 1, 1, t_step, pad)


    x_diff_grid = grad_i_x(x_step, kernelx, 1)
    y_diff_grid = grad_i_x(y_step, kernely, 1)
    z_diff_grid = grad_i_x(z_step, kernelz, 1)

    u_x = u_x / x_diff_grid
    u_y = u_y / y_diff_grid
    u_z = u_z / z_diff_grid

    v_x = v_x / x_diff_grid
    v_y = v_y / y_diff_grid
    v_z = v_z / z_diff_grid

    w_x = w_x / x_diff_grid
    w_y = w_y / y_diff_grid
    w_z = w_z / z_diff_grid



    if kernel5:
        u_ = u[:, :, 2:-2, 2:-2, 2:-2]
        v_ = v[:, :, 2:-2, 2:-2, 2:-2]
        w_ = w[:, :, 2:-2, 2:-2, 2:-2]
    else:
        u_ = u[:, :, 1:-1, 1:-1, 1:-1]
        v_ = v[:, :, 1:-1, 1:-1, 1:-1]
        w_ = w[:, :, 1:-1, 1:-1, 1:-1]
    # X分量
    if cengliu:
        uu_x = u_[1:-1, :, :, :, :] * grad_i_x(u[1:-1, :, :, :, :], kernelx, 1)
        uv_y = v_[1:-1, :, :, :, :] * grad_i_x(u[1:-1, :, :, :, :], kernely, 1)
        uw_z = w_[1:-1, :, :, :, :] * grad_i_x(u[1:-1, :, :, :, :], kernelz, 1)
    else:
        uu_x = grad_i_x(u[1:-1, :, :, :, :] * u[1:-1, :, :, :, :], kernelx, 1)
        uv_y = grad_i_x(v[1:-1, :, :, :, :] * u[1:-1, :, :, :, :], kernely, 1)
        uw_z = grad_i_x(w[1:-1, :, :, :, :] * u[1:-1, :, :, :, :], kernelz, 1)
    u_t = cal_t_diff_pad(u_) / t_step
    uu_x = uu_x / x_diff_grid
    uv_y = uv_y / y_diff_grid
    uw_z = uw_z / z_diff_grid
    # y分量
    if cengliu:
        vu_x = u_[1:-1, :, :, :, :] * grad_i_x(v[1:-1, :, :, :, :], kernelx, 1)
        vv_y = v_[1:-1, :, :, :, :] * grad_i_x(v[1:-1, :, :, :, :], kernely, 1)
        vw_z = w_[1:-1, :, :, :, :] * grad_i_x(v[1:-1, :, :, :, :], kernelz, 1)
    else:
        vu_x = grad_i_x(u[1:-1, :, :, :, :] * v[1:-1, :, :, :, :], kernelx, 1)
        vv_y = grad_i_x(v[1:-1, :, :, :, :] * v[1:-1, :, :, :, :], kernely, 1)
        vw_z = grad_i_x(w[1:-1, :, :, :, :] * v[1:-1, :, :, :, :], kernelz, 1)

    v_t = cal_t_diff(v_) / t_step
    vu_x = vu_x / x_diff_grid
    vv_y = vv_y / y_diff_grid
    vw_z = vw_z / z_diff_grid
    # Z分量
    if cengliu:
        wu_x = u_[1:-1, :, :, :, :] * grad_i_x(w[1:-1, :, :, :, :], kernelx, 1)
        wv_y = v_[1:-1, :, :, :, :] * grad_i_x(w[1:-1, :, :, :, :], kernely, 1)
        ww_z = w_[1:-1, :, :, :, :] * grad_i_x(w[1:-1, :, :, :, :], kernelz, 1)
    else:
        wu_x = grad_i_x(u[1:-1, :, :, :, :] * w[1:-1, :, :, :, :], kernelx, 1)
        wv_y = grad_i_x(v[1:-1, :, :, :, :] * w[1:-1, :, :, :, :], kernely, 1)
        ww_z = grad_i_x(w[1:-1, :, :, :, :] * w[1:-1, :, :, :, :], kernelz, 1)

    w_t = cal_t_diff(w_) / t_step
    wu_x = wu_x / x_diff_grid
    wv_y = wv_y / y_diff_grid
    ww_z = ww_z / z_diff_grid

    # 求二阶偏导
    if kernel5:
        x_diff_grid = x_diff_grid[:, :, 2:-2, 2:-2, 2:-2]
        y_diff_grid = y_diff_grid[:, :, 2:-2, 2:-2, 2:-2]
        z_diff_grid = z_diff_grid[:, :, 2:-2, 2:-2, 2:-2]
    else:
        x_diff_grid = x_diff_grid[:, :, 1:-1, 1:-1, 1:-1]
        y_diff_grid = y_diff_grid[:, :, 1:-1, 1:-1, 1:-1]
        z_diff_grid = z_diff_grid[:, :, 1:-1, 1:-1, 1:-1]
    u_xx = F.conv3d(u_x, kernelx, bias=None) / x_diff_grid
    u_yy = F.conv3d(u_y, kernely, bias=None) / y_diff_grid
    u_zz = F.conv3d(u_z, kernelz, bias=None) / z_diff_grid
    if pad:
        u_xx = RepPad(u_xx)
        u_yy = RepPad(u_yy)
        u_zz = RepPad(u_zz)
    v_xx = F.conv3d(v_x, kernelx, bias=None) / x_diff_grid
    v_yy = F.conv3d(v_y, kernely, bias=None) / y_diff_grid
    v_zz = F.conv3d(v_z, kernelz, bias=None) / z_diff_grid
    if pad:
        v_xx = RepPad(v_xx)
        v_yy = RepPad(v_yy)
        v_zz = RepPad(v_zz)
    w_xx = F.conv3d(w_x, kernelx, bias=None) / x_diff_grid
    w_yy = F.conv3d(w_y, kernely, bias=None) / y_diff_grid
    w_zz = F.conv3d(w_z, kernelz, bias=None) / z_diff_grid

    if kernel5:
        u_x = u_x[:, :, 2:-2, 2:-2, 2:-2]
        v_y = v_y[:, :, 2:-2, 2:-2, 2:-2]
        w_z = w_z[:, :, 2:-2, 2:-2, 2:-2]

        u_t = u_t[:, :, 2:-2, 2:-2, 2:-2]
        v_t = v_t[:, :, 2:-2, 2:-2, 2:-2]
        w_t = w_t[:, :, 2:-2, 2:-2, 2:-2]

        uu_x = uu_x[:, :, 2:-2, 2:-2, 2:-2]
        uv_y = uv_y[:, :, 2:-2, 2:-2, 2:-2]
        uw_z = uw_z[:, :, 2:-2, 2:-2, 2:-2]

        vu_x = vu_x[:, :, 2:-2, 2:-2, 2:-2]
        vv_y = vv_y[:, :, 2:-2, 2:-2, 2:-2]
        vw_z = vw_z[:, :, 2:-2, 2:-2, 2:-2]

        wu_x = wu_x[:, :, 2:-2, 2:-2, 2:-2]
        wv_y = wv_y[:, :, 2:-2, 2:-2, 2:-2]
        ww_z = ww_z[:, :, 2:-2, 2:-2, 2:-2]


    else:
        u_x = u_x[:, :, 1:-1, 1:-1, 1:-1]
        v_y = v_y[:, :, 1:-1, 1:-1, 1:-1]
        w_z = w_z[:, :, 1:-1, 1:-1, 1:-1]

        u_t = u_t[:, :, 1:-1, 1:-1, 1:-1]
        v_t = v_t[:, :, 1:-1, 1:-1, 1:-1]
        w_t = w_t[:, :, 1:-1, 1:-1, 1:-1]

        uu_x = uu_x[:, :, 1:-1, 1:-1, 1:-1]
        uv_y = uv_y[:, :, 1:-1, 1:-1, 1:-1]
        uw_z = uw_z[:, :, 1:-1, 1:-1, 1:-1]

        vu_x = vu_x[:, :, 1:-1, 1:-1, 1:-1]
        vv_y = vv_y[:, :, 1:-1, 1:-1, 1:-1]
        vw_z = vw_z[:, :, 1:-1, 1:-1, 1:-1]

        wu_x = wu_x[:, :, 1:-1, 1:-1, 1:-1]
        wv_y = wv_y[:, :, 1:-1, 1:-1, 1:-1]
        ww_z = ww_z[:, :, 1:-1, 1:-1, 1:-1]

    p_x = -density*(u_t + (uu_x + uv_y + uw_z)  - viscosity * (u_xx + u_yy  +u_zz))
    p_y = -density*(v_t + (vu_x + vv_y + vw_z)  - viscosity * (v_xx + v_yy + v_zz))
    p_z = -density*(w_t + (wu_x + wv_y + ww_z)  - viscosity * (w_xx + w_yy + w_zz))

    p_dif_pre=torch.cat([p_x,p_y,p_z],dim=1)

    return p_dif_pre

def cal_P_by_NS_queen2(u, v, w,x_step, y_step, z_step, t_step,density=1, viscosity=5e-5,pad=0,kernel5=0,cengliu=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
    # Sobel算子

    # kernelx = torch.tensor([[[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]],
    #                         [[-2.,-2., 4.],
    #                          [-4., -4., 8.],
    #                          [-2., -2., 4.]],
    #                         [[-1., -1., 2.],
    #                          [-2., -2., 4.],
    #                          [-1., -1., 2.]]])
    if kernel5:
        kernelx = torch.tensor([[[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [2., -54., 0., 54., -2.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [1., -27., 0., 27., -1.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]],

                                [[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]]
                                ])
        kernelx = kernelx / torch.sum(abs(kernelx))
    else:
        kernelx = torch.tensor([[[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-1., 0., 1.],
                                 [-0., 0., 0.]],
                                [[-0., 0., 0.],
                                 [-0., 0., 0.],
                                 [-0., 0., 0.]]])

        kernelx = kernelx / torch.sum(abs(kernelx))
    # kernelx = torch.tensor([[[0., 0., 0.],
    #                          [0., 0., 0.],
    #                          [0., 0., 0.]],
    #                         [[0., 0., 0.],
    #                          [-1., 0.,1.],
    #                          [0., 0., 0.]],
    #                         [[0., 0., 0.],
    #                          [0., 0., 0.],
    #                          [0., 0., 0.]]]) / 2
    kernely = kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    if kernel5:
        kernelx = kernelx.reshape(1, 1, 5, 5, 5).cuda()
        kernely = kernely.reshape(1, 1, 5, 5, 5).cuda()
        kernelz = kernelz.reshape(1, 1, 5, 5, 5).cuda()
    else:
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).cuda()
        kernely = kernely.reshape(1, 1, 3, 3, 3).cuda()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).cuda()
    dx = grad_i_x(x_step, kernelx, 1)
    dy = grad_i_x(y_step, kernely, 1)
    dz = grad_i_x(z_step, kernelz, 1)
    vel=torch.cat([u,v,w],dim=1)[0]
    vel2=torch.cat([u,v,w],dim=1)[1]
    x=x_step
    y=y_step
    z=z_step
    u1=u[0]
    v1=v[0]
    w1=w[0]

    u2=u[1]
    v2=v[1]
    w2=w[1]

    ua=(u1+u2)/2
    va=(v1+v2)/2
    wa=(w1+w2)/2

    xa= (x+ua*t_step)[0][0]
    ya= (y+va*t_step)[0][0]
    za= (z+wa*t_step)[0][0]

    from scipy.interpolate import griddata


    u2_line=u2.reshape(-1,1).cpu().numpy()
    v2_line=v2.reshape(-1,1).cpu().numpy()
    w2_line=w2.reshape(-1,1).cpu().numpy()
    u2=u2.cpu().numpy()
    v2=v2.cpu().numpy()
    w2=w2.cpu().numpy()


    x_init = x.reshape(-1, 1)
    y_init = y.reshape(-1, 1)
    z_init = z.reshape(-1, 1)
    points = np.concatenate([x_init.cpu().numpy(), y_init.cpu().numpy(), z_init.cpu().numpy()], axis=1)

    u2a=  griddata(points,u2_line,  (xa.cpu().numpy(),ya.cpu().numpy(),  za.cpu().numpy()), method='nearest').transpose(3, 0, 1, 2)[0]
    v2a = griddata(points, v2_line, (xa.cpu().numpy(), ya.cpu().numpy(), za.cpu().numpy()), method='nearest').transpose(3, 0, 1, 2)[0]
    w2a = griddata(points, w2_line, (xa.cpu().numpy(), ya.cpu().numpy(), za.cpu().numpy()), method='nearest').transpose(3, 0, 1, 2)[0]

    px=density*((u2a-u1[0].cpu().numpy())/t_step.cpu().numpy())
    plt.subplot(131)
    plt.imshow(px[:,10,:],cmap='jet',vmax=1,vmin=-1)
    plt.colorbar(shrink=0.5)
#    return p_dif_pre
def test_Beltrami_flow():
    x = np.linspace(-1, 1,num=33 )
    y = np.linspace(-1, 1, num=33)[::-1]
    z =np.linspace(-1, 1 , num=33)
    tg = np.linspace(0, 1,num=33 )
    density = 1.
    viscosity = 1.
    t_step = (tg[-1] - tg[0]) / (tg.shape[0] - 1)
    [yg, zg, xg] = np.meshgrid(y[1:], z[1:], x[1:])
    yg = torch.from_numpy(yg).cuda()
    zg = torch.from_numpy(zg).cuda()
    xg = torch.from_numpy(xg).cuda()
    tg = torch.from_numpy(tg).cuda()
    u, v, w, p = gen_flow(xg, yg, zg, tg)
    plt.subplot(221)
    plt.imshow(u[0,:,0,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(v[0,:,0,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(w[0,:,0,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(p[0,:,0,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.show()
    u=u.unsqueeze(dim=1)
    v=v.unsqueeze(dim=1)
    w=w.unsqueeze(dim=1)
    p=p.unsqueeze(dim=1)
    xg=xg.unsqueeze(dim=0).unsqueeze(dim=0).float()
    yg=yg.unsqueeze(dim=0).unsqueeze(dim=0).float()
    zg=zg.unsqueeze(dim=0).unsqueeze(dim=0).float()
    #cal_NS_residual_cengliu(u, v, w, p, x_step, y_step, z_step, t_step, density, viscosity)
    cal_NS_residual_vv_rotate(u, v, w, xg, yg, zg, t_step,density,viscosity)
    # pow_cont, pow_nsX, pow_nsY, pow_nsZ=cal_NS_residual_tuanliu(u, v, w, p
    #                                                             , xg
    #                                                             , yg
    #                                                             , zg,
    #                                                             t_step, density, viscosity,cengliu=0)

def test_taylor_green_flow():
    x = np.linspace(-np.pi*2, np.pi*2,num=65 )
    y = np.linspace(-np.pi*2, np.pi*2, num=65)
    z = np.linspace(-np.pi*2, np.pi*2 , num=65)
    tg = np.linspace(0, 1,num=11 )
    density = 1.
    viscosity = 1
    t_step = (tg[-1] - tg[0]) / (tg.shape[0] - 1)
    [yg, zg, xg] = np.meshgrid(y[1:], z[1:], x[1:])
    yg = torch.from_numpy(yg).cuda()
    zg = torch.from_numpy(zg).cuda()
    xg = torch.from_numpy(xg).cuda()
    tg = torch.from_numpy(tg).cuda()
    u, v, w, p = gen_taylor_green_flow(xg, yg, zg, tg,viscosity)
    plt.subplot(221)
    plt.imshow(u[0,:,32,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(v[0,:,32,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(w[0,:,32,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(p[0,:,32,:].cpu(),cmap='jet')
    plt.colorbar()
    plt.show()
    u=u.unsqueeze(dim=1)
    v=v.unsqueeze(dim=1)
    w=w.unsqueeze(dim=1)
    p=p.unsqueeze(dim=1)
    xg=xg.unsqueeze(dim=0).unsqueeze(dim=0).float()
    yg=yg.unsqueeze(dim=0).unsqueeze(dim=0).float()
    zg=zg.unsqueeze(dim=0).unsqueeze(dim=0).float()
    #cal_NS_residual_cengliu(u, v, w, p, x_step, y_step, z_step, t_step, density, viscosity)
    cal_NS_residual_vv_rotate(u, v, w, xg, yg, zg, t_step,density,viscosity)
    # pow_cont, pow_nsX, pow_nsY, pow_nsZ=cal_NS_residual_tuanliu(u, v, w, p
    #                                                             , xg
    #                                                             , yg
    #                                                             , zg,
    #                                                             t_step, density, viscosity,cengliu=0)
    #print(pow_cont)
def test_JHTDB():
    flo_folder='H:\三维流场重建\数据集\iso_tropic_new\\fine\pth'
    flo_paths=[]
    for name in os.listdir(flo_folder):
        if name.endswith('pth'):
            flo_paths.append(os.path.join(flo_folder,name))
    flows_t=[]
    for path in flo_paths:
        flo=torch.load(path)
        flows_t.append(flo['v_p'])
    flows_t=torch.stack(flows_t)

    # u=  torch.from_numpy(flows_t.numpy()[:3, 0, ::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=1)
    # v = torch.from_numpy(flows_t.numpy()[:3, 1, ::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=1)
    # w = torch.from_numpy(flows_t.numpy()[:3, 2, ::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=1)
    # p = torch.from_numpy(flows_t.numpy()[:3, 3, ::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=1)
    # x_step= torch.from_numpy(flo['x_coord'][::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    # y_step= torch.from_numpy(flo['y_coord'][::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    # z_step= torch.from_numpy(flo['z_coord'][::2, ::-2, ::2].copy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
    # t_step= torch.tensor(flo['t_step']).float().cuda()
    # _, _, D, H, W = p.shape

    u=  torch.from_numpy(flows_t.numpy()[:, 0, :64, :64:1, :64:].copy()).cuda().unsqueeze(dim=1).double()
    v = torch.from_numpy(flows_t.numpy()[:, 1, :64, :64:1, :64:].copy()).cuda().unsqueeze(dim=1).double()
    w = torch.from_numpy(flows_t.numpy()[:, 2, :64, :64:1, :64:].copy()).cuda().unsqueeze(dim=1).double()
    p = torch.from_numpy(flows_t.numpy()[:, 3, :64, :64:1, :64:].copy()).cuda().unsqueeze(dim=1).double()
    x_step= torch.from_numpy(flo['x_coord'][:64, :64:1, :64:].copy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0).double()
    y_step= torch.from_numpy(flo['y_coord'][:64, :64:1, :64:].copy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0).double()
    z_step= torch.from_numpy(flo['z_coord'][:64, :64:1, :64:].copy()).cuda().unsqueeze(dim=0).unsqueeze(dim=0).double()
    t_step= 1*torch.tensor(flo['t_step']).float().cuda()
    _,_,D,H,W=p.shape

    # dert_t_1 = torch.ones_like(u) * 0.0065 * 0.45
    # dert_t_0 = torch.zeros_like(u)
    # dert_t_move_u = -torch.cat([dert_t_0, dert_t_1,dert_t_1], dim=1)*0.5
    # dert_t_move_v = -torch.cat([dert_t_1, dert_t_0, dert_t_1], dim=1)*0.5
    # dert_t_move_w = -torch.cat([dert_t_1, dert_t_1, dert_t_0], dim=1)*0.5
    # dert_t_move_p = -torch.cat([dert_t_1, dert_t_1, dert_t_1], dim=1)*0.5
    #
    # u_move=forward_warp3d(u , dert_t_move_u)
    # v_move = forward_warp3d(v, dert_t_move_v)
    # w_move = forward_warp3d(w, dert_t_move_w)
    # p_move = forward_warp3d(p, dert_t_move_p)


    # plt.subplot(221)
    # plt.imshow(u[0,0,:,10,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(v[0,0,:,10,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(w[0,0,:,10,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(p2[0,0,:,10,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.show()
    # plt.subplot(221)
    # plt.imshow(u[0,0,10,:,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(v[0,0,10,:,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(w[0,0,10,:,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(p[0,0,10,:,:].cpu(),cmap='jet')
    # plt.colorbar()
    # plt.show()
    density = 1.0
    for key in flo.keys():
        if key=='density':
            density=flo['density']
    viscosity=flo['viscosity']
    #print(x_step[0,0,0,1,1].cpu().numpy())
    #p_dif_pre=cal_P_by_NS(u_move, v_move, w_move,x_step,y_step,z_step,t_step,density,viscosity)
    #cal_P_by_NS_queen2(u, v, w,x_step,y_step,z_step,t_step,density,viscosity)
    # p_dif_mean=torch.mean(p_dif_pre,dim=0)
    # plt.subplot(132)
    # plt.imshow(p_dif_mean.cpu()[0,:,10,:],cmap='jet',vmax=1,vmin=-1)
    # plt.colorbar(shrink=0.5)
    #cal_NS_residual_vv(u, v, w, x_step, y_step, z_step, t_step, density, viscosity, pad=0, cengliu=1,kernel5=0)
    #cal_NS_residual_vv(u,v,w,x_step,y_step,z_step,t_step)

    # pow_cont, pow_nsX, pow_nsY, pow_nsZ = cal_NS_residual_tuanliu(u_move, v_move, w_move
    #                                                               , p_move, x_step, y_step, z_step,
    #                                                               t_step, density,
    #                                                               viscosity,cengliu=0)
    cal_NS_residual_vv_rotate(u, v, w, x_step ,y_step, z_step, t_step, density, viscosity)
    #pow_cont, pow_nsX, pow_nsY, pow_nsZ = cal_NS_residual_tuanliu(u, v, w
    #                                                               , p, x_step, y_step, z_step,
    #                                                               t_step, density,
    #                                                               viscosity,cengliu=0)
#test_taylor_green_flow()
#test_JHTDB()
#test_Beltrami_flow()



