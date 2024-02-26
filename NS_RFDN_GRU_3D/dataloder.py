import random
import os
import numpy as np
from plot_velocity_fan import read_flow
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from N_S_loss_Fan_init_coord import grad_i, grad_i_x
from torch_gauuskernel import get_gaussian_kernel, get_mean_kernel

'''
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor
'''
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
    u_t_end = u[-1] - u[-2]
    u_t.append(u_t_end)
    u_t = torch.stack(u_t, dim=0)
    return u_t


train_transformer = transforms.Compose([transforms.ToTensor()])
eval_transformer = transforms.Compose([transforms.ToTensor()])
# def cal_P_by_NS(u, v, w,x_step, y_step, z_step, t_step,density=1, viscosity=5e-5,pad=1,kernel5=0,cengliu=1):
#     RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
#     # Sobel算子
#     kernelx = torch.tensor([[[-1., 0., 1.],
#                              [-2., 0., 2.],
#                              [-1., 0., 1.]],
#                             [[-2., 0., 2.],
#                              [-4., 0., 4.],
#                              [-2., 0., 2.]],
#                             [[-1., 0., 1.],
#                              [-2., 0., 2.],
#                              [-1., 0., 1.]]])
#     kernelx = kernelx / torch.sum(abs(kernelx))
#     kernely = kernelx.transpose(1, 2)
#     kernelz = kernelx.transpose(2, 0)
#
#     kernelx = kernelx.reshape(1, 1, 3, 3, 3).cuda().double()
#     kernely = kernely.reshape(1, 1, 3, 3, 3).cuda().double()
#     kernelz = kernelz.reshape(1, 1, 3, 3, 3).cuda().double()
#     # 求一阶偏导
#     u_t = cal_t_diff_pad(u) / t_step
#     v_t = cal_t_diff_pad(v) / t_step
#     w_t = cal_t_diff_pad(w) / t_step
#
#     u_x, u_y, u_z = grad_i(u, 1, 1, 1, pad)
#     v_x, v_y, v_z = grad_i(v, 1, 1, 1, pad)
#     w_x, w_y, w_z = grad_i(w, 1, 1, 1, pad)
#
#
#     x_diff_grid = grad_i_x(x_step, kernelx, 1)
#     y_diff_grid = grad_i_x(y_step, kernely, 1)
#     z_diff_grid = grad_i_x(z_step, kernelz, 1)
#
#     # X分量
#     if cengliu:
#         uu_x = u * u_x
#         uv_y = v * u_y
#         uw_z = w * u_z
#     else:
#         uu_x = grad_i_x(u * u, kernelx, 1)
#         uv_y = grad_i_x(v * u, kernely, 1)
#         uw_z = grad_i_x(w * u, kernelz, 1)
#
#     uu_x = uu_x / x_diff_grid
#     uv_y = uv_y / y_diff_grid
#     uw_z = uw_z / z_diff_grid
#     # y分量
#     if cengliu:
#         vu_x = u * v_x
#         vv_y = v * v_y
#         vw_z = w * v_z
#     else:
#         vu_x = grad_i_x(u * v, kernelx, 1)
#         vv_y = grad_i_x(v * v, kernely, 1)
#         vw_z = grad_i_x(w * v, kernelz, 1)
#
#     vu_x = vu_x / x_diff_grid
#     vv_y = vv_y / y_diff_grid
#     vw_z = vw_z / z_diff_grid
#     # Z分量
#     if cengliu:
#         wu_x = u * w_x
#         wv_y = v * w_y
#         ww_z = w * w_z
#     else:
#         wu_x = grad_i_x(u * w, kernelx, 1)
#         wv_y = grad_i_x(v * w, kernely, 1)
#         ww_z = grad_i_x(w * w, kernelz, 1)
#
#     wu_x = wu_x / x_diff_grid
#     wv_y = wv_y / y_diff_grid
#     ww_z = ww_z / z_diff_grid
#
#     # 求二阶偏导
#
#     u_xx = grad_i_x(u_x,kernelx,1) / x_diff_grid
#     u_yy = grad_i_x(u_y,kernely,1) / y_diff_grid
#     u_zz = grad_i_x(u_z,kernelz,1) / z_diff_grid
#
#     v_xx = grad_i_x(v_x,kernelx,1) / x_diff_grid
#     v_yy = grad_i_x(v_y,kernely,1) / y_diff_grid
#     v_zz = grad_i_x(v_z,kernelz,1) / z_diff_grid
#
#     w_xx = grad_i_x(w_x,kernelx,1) / x_diff_grid
#     w_yy = grad_i_x(w_x,kernelx,1) / x_diff_grid
#     w_zz = grad_i_x(w_x,kernelx,1) / x_diff_grid
#
#     p_x = -density*(u_t + (uu_x + uv_y + uw_z)  - viscosity * (u_xx + u_yy  +u_zz))
#     p_y = -density*(v_t + (vu_x + vv_y + vw_z)  - viscosity * (v_xx + v_yy + v_zz))
#     p_z = -density*(w_t + (wu_x + wv_y + ww_z)  - viscosity * (w_xx + w_yy + w_zz))
#
#
#
#     return p_x, p_y, p_z

class FACESDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    # Note that the first directory is train, the second directory is label
    def __init__(self, label_data_dir,input_t_length,transform,is_train,downsample_form,down_sample=2,time_interp=True):  ########################### Here add a new parameter "blur_data_dir"
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.label_filenames=[]
        self.is_train=is_train
        self.time_interp=time_interp
        self.down_sample=down_sample
        total_names=os.listdir(label_data_dir)
        mini_group_num=int(len(total_names)/input_t_length)
        for i in range(mini_group_num):
            mini_group=[]
            for j in range(input_t_length):
                mini_group.append(os.path.join(label_data_dir,total_names[i*input_t_length+j]))
            self.label_filenames.append(mini_group)
        self.transform = transform
        total_group_num=len(self.label_filenames)
        self.down_form=downsample_form

    def __len__(self):
        # return size of dataset
        return len(self.label_filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed blur image
            label: (Tensor) transformed original image
        """
        #input
        kernelx = (torch.tensor([[[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]],
                                [[-2., 0., 2.],
                                 [-4., 0., 4.],
                                 [-2., 0., 2.]],
                                [[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]])/32).float()
        pad_num = 1
        kernely = kernelx.transpose(1, 2)
        kernelz = kernelx.transpose(2, 0)
        kernelx = kernelx.reshape(1, 1, 3, 3, 3).cuda()
        kernely = kernely.reshape(1, 1, 3, 3, 3).cuda()
        kernelz = kernelz.reshape(1, 1, 3, 3, 3).cuda()
        # flows_t = []
        # input_paths=self.input_filenames[idx]
        # for path in input_paths:
        #     flo = torch.load(path)
        #     flows_t.append(flo['v_p'])
        # flows_t = torch.stack(flows_t)
        # u = torch.from_numpy(flows_t.numpy()[:, 0:1, :, :, :]).cuda()
        # v = torch.from_numpy(flows_t.numpy()[:, 1:2, :, :, :]).cuda()
        # w = torch.from_numpy(flows_t.numpy()[:, 2:3, :, :, :]).cuda()
        # p = torch.from_numpy(flows_t.numpy()[:, 3:4, :, :, :]).cuda()
        # x_step = torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda()
        # y_step = torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda()
        # z_step = torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda()
        # x_diff_grid = abs(grad_i_x(x_step.float(), kernelx, 1, pad=pad_num))
        # y_diff_grid = abs(grad_i_x(y_step.float(), kernely, 1, pad=pad_num))
        # z_diff_grid = abs(grad_i_x(z_step.float(), kernelz, 1, pad=pad_num))
        #
        # u_x = grad_i_x(u, kernelx, 1, pad=pad_num) / x_diff_grid
        # u_y = grad_i_x(u, kernely, 1, pad=pad_num) / y_diff_grid
        # u_z = grad_i_x(u, kernelz, 1, pad=pad_num) / z_diff_grid
        #
        # v_x = grad_i_x(v, kernelx, 1, pad=pad_num) / x_diff_grid
        # v_y = grad_i_x(v, kernely, 1, pad=pad_num) / y_diff_grid
        # v_z = grad_i_x(v, kernelz, 1, pad=pad_num) / z_diff_grid
        #
        # w_x = grad_i_x(w, kernelx, 1, pad=pad_num) / x_diff_grid
        # w_y = grad_i_x(w, kernely, 1, pad=pad_num) / y_diff_grid
        # w_z = grad_i_x(w, kernelz, 1, pad=pad_num) / z_diff_grid
        # oumiga_x = 0.5*(w_y - v_z)
        # oumiga_y = 0.5*(u_z - w_x)
        # oumiga_z = 0.5*(v_x - u_y)
        # has_density=0
        # # for key in flo.keys():
        # #     if key=='density':
        # #         has_density=1
        # #         p_x,p_y,p_z=cal_P_by_NS(u, v, w,x_step, y_step, z_step, flo['t_step']*2,
        # #                                 density=flo['density'], viscosity=flo['viscosity'],
        # #                                 pad=1,kernel5=0,cengliu=0)
        # # if not has_density:
        # #     p_x, p_y, p_z = cal_P_by_NS(u, v, w, x_step, y_step, z_step, flo['t_step']*2,
        # #                                 density=1, viscosity=flo['viscosity'],
        # #                                 pad=1, kernel5=0, cengliu=0)
        # # plt.subplot(221)
        # #         # plt.imshow(p_x[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # #         # plt.subplot(222)
        # #         # plt.imshow(4*p_x1[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # #         # plt.subplot(223)
        # #         # plt.imshow(p_y[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # #         # plt.subplot(224)
        # #         # plt.imshow(4*p_y1[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # #         # plt.savefig('D:\desktop\\input.png')
        #
        #
        # v_oumiga1 = torch.cat([u.cpu(), v.cpu(), w.cpu(), oumiga_x.cpu(), oumiga_y.cpu(), oumiga_z.cpu()], dim=1)
        # input_v_dp_dict = {}
        # input_v_dp_dict['v_oumiga'] = v_oumiga1.float()
        # input_v_dp_dict['x_grid'] = x_step.cpu().float()
        # input_v_dp_dict['y_grid'] = y_step.cpu().float()
        # input_v_dp_dict['z_grid'] = z_step.cpu().float()
        # input_v_dp_dict['density'] = 1
        # input_v_dp_dict['viscosity'] = flo['viscosity']
        # input_v_dp_dict['t_step']=flo['t_step']





        # label
        label_flows_t = []
        label_paths=self.label_filenames[idx]
        for path in label_paths:
            flo = torch.load(path)
            if isinstance(flo['v_p'],np.ndarray):
                label_flows_t.append(torch.from_numpy(flo['v_p']))
            elif isinstance(flo['v_p'],torch.Tensor):
                label_flows_t.append(flo['v_p'])
        label_flows_t = torch.stack(label_flows_t)
        
        _,_,_,_,w_=label_flows_t.shape
        label_flows_t_ = label_flows_t.cpu().numpy()
        if w_>100:
            if not self.is_train:
                u = torch.from_numpy(label_flows_t.numpy()[:, 0:1, :, :, :]).cuda().float()
                v = torch.from_numpy(label_flows_t.numpy()[:, 1:2, :, :, :]).cuda().float()
                w = torch.from_numpy(label_flows_t.numpy()[:, 2:3, :, :,:]).cuda().float()
                p = torch.from_numpy(label_flows_t.numpy()[:, 3:4, :, :, :]).cuda().float()
                if isinstance(flo['x_coord'],np.ndarray):
                    x_step = torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,:].cuda().float()
                    y_step = torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,:].cuda().float()
                    z_step = torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,:].cuda().float()
                else:
                    x_step=flo['x_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,:].cuda().float()
                    y_step = flo['y_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda().float()
                    z_step = flo['z_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda().float()
            else:
                u = torch.from_numpy(label_flows_t.numpy()[:, 0:1, :, 2:, 2:130]).cuda().float()
                v = torch.from_numpy(label_flows_t.numpy()[:, 1:2, :, 2:, 2:130]).cuda().float()
                w = torch.from_numpy(label_flows_t.numpy()[:, 2:3, :, 2:, 2:130]).cuda().float()
                p = torch.from_numpy(label_flows_t.numpy()[:, 3:4, :, 2:, 2:130]).cuda().float()
                if isinstance(flo['x_coord'],np.ndarray):
                    x_step = torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, 2:, 2:130].cuda().float()
                    y_step = torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, 2:, 2:130].cuda().float()
                    z_step = torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, 2:, 2:130].cuda().float()
                else:
                    x_step=flo['x_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, 2:, 2:130].cuda().float()
                    y_step = flo['y_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, 2:, 2:130].cuda().float()
                    z_step = flo['z_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, 2:, 2:130].cuda().float()
        else:
            u = torch.from_numpy(label_flows_t.numpy()[:, 0:1, :, :, :]).cuda().float()
            v = torch.from_numpy(label_flows_t.numpy()[:, 1:2, :, :, :]).cuda().float()
            w = torch.from_numpy(label_flows_t.numpy()[:, 2:3, :, :, :]).cuda().float()
            p = torch.from_numpy(label_flows_t.numpy()[:, 3:4, :, :, :]).cuda().float()
            if isinstance(flo['x_coord'], np.ndarray):
                x_step = torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,
                         :].cuda().float()
                y_step = torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,
                         :].cuda().float()
                z_step = torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :,
                         :].cuda().float()
            else:
                x_step = flo['x_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda().float()
                y_step = flo['y_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda().float()
                z_step = flo['z_coord'].unsqueeze(dim=0).unsqueeze(dim=0)[:, :, :, :, :].cuda().float()
        x_diff_grid = abs(grad_i_x(x_step.float(), kernelx, 1, pad=pad_num)).float()
        y_diff_grid = abs(grad_i_x(y_step.float(), kernely, 1, pad=pad_num)).float()
        z_diff_grid = abs(grad_i_x(z_step.float(), kernelz, 1, pad=pad_num)).float()

        u_x = grad_i_x(u, kernelx, 1, pad=pad_num) / x_diff_grid
        u_y = grad_i_x(u, kernely, 1, pad=pad_num) / y_diff_grid
        u_z = grad_i_x(u, kernelz, 1, pad=pad_num) / z_diff_grid

        v_x = grad_i_x(v, kernelx, 1, pad=pad_num) / x_diff_grid
        v_y = grad_i_x(v, kernely, 1, pad=pad_num) / y_diff_grid
        v_z = grad_i_x(v, kernelz, 1, pad=pad_num) / z_diff_grid

        w_x = grad_i_x(w, kernelx, 1, pad=pad_num) / x_diff_grid
        w_y = grad_i_x(w, kernely, 1, pad=pad_num) / y_diff_grid
        w_z = grad_i_x(w, kernelz, 1, pad=pad_num) / z_diff_grid
        oumiga_x = 0.5*(w_y - v_z)
        oumiga_y = 0.5*(u_z - w_x)
        oumiga_z = 0.5*(v_x - u_y)
        has_density=0
        # for key in flo.keys():
        #     if key=='density':
        #         has_density=1
        #         p_x,p_y,p_z=cal_P_by_NS(u, v, w,x_step, y_step, z_step, flo['t_step'],
        #                                 density=flo['density'], viscosity=flo['viscosity'],
        #                                 pad=1,kernel5=0,cengliu=0)
        # if not has_density:
        #     p_x, p_y, p_z = cal_P_by_NS(u, v, w, x_step, y_step, z_step, flo['t_step'],
        #                                 density=1, viscosity=flo['viscosity'],
        #                                 pad=1, kernel5=0, cengliu=0)
        v_oumiga2 = torch.cat([u.cpu(), v.cpu(), w.cpu(), oumiga_x.cpu(), oumiga_y.cpu(), oumiga_z.cpu()], dim=1)
        label_v_dp_dict = {}
        label_v_dp_dict['name'] = label_paths
        label_v_dp_dict['v_oumiga'] = v_oumiga2.float()
        label_v_dp_dict['x_grid'] = x_step.cpu().float()
        label_v_dp_dict['y_grid'] = y_step.cpu().float()
        label_v_dp_dict['z_grid'] = z_step.cpu().float()
        label_v_dp_dict['density'] = 1
        label_v_dp_dict['viscosity'] = flo['viscosity']
        label_v_dp_dict['t_step']=flo['t_step']
        matplotlib.use('TkAgg')
        # plt.subplot(221)
        # plt.imshow(p_x[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # plt.subplot(222)
        # plt.imshow(4*p_x2[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # plt.subplot(223)
        # plt.imshow(p_y[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # plt.subplot(224)
        # plt.imshow(4*p_y2[0,0,:,10,:].cpu(),vmax=0.8,vmin=-0.8,cmap='jet')
        # plt.savefig('D:\desktop\\label.png')
        if self.down_form=='gauss':
            down_kernel=get_gaussian_kernel(channels=v_oumiga2.shape[1])
        elif self.down_form=='mean':
            down_kernel=get_mean_kernel(self.down_sample)
        elif self.down_form is None:
            down_kernel=None

        if down_kernel is not None:
            if self.time_interp:
                input_v_dp_dict={}
                input_v_dp_dict['name']=label_paths[::2]
                input_v_dp_dict['v_oumiga'] = down_kernel(v_oumiga2.float()[::2])
                input_v_dp_dict['x_grid']=  down_kernel(x_step.cpu().float()[::2])
                input_v_dp_dict['y_grid'] = down_kernel(y_step.cpu().float()[::2])
                input_v_dp_dict['z_grid'] = down_kernel(z_step.cpu().float()[::2])
                input_v_dp_dict['density'] = 1
                input_v_dp_dict['viscosity'] = flo['viscosity']
                input_v_dp_dict['t_step'] = flo['t_step']*2
            else:
                input_v_dp_dict={}
                input_v_dp_dict['name']=label_paths[::]
                input_v_dp_dict['v_oumiga'] = down_kernel(v_oumiga2.float()[::])
                input_v_dp_dict['x_grid']=  down_kernel(x_step.cpu().float()[::])
                input_v_dp_dict['y_grid'] = down_kernel(y_step.cpu().float()[::])
                input_v_dp_dict['z_grid'] = down_kernel(z_step.cpu().float()[::])
                input_v_dp_dict['density'] = 1
                input_v_dp_dict['viscosity'] = flo['viscosity']
                input_v_dp_dict['t_step'] = flo['t_step']

            return input_v_dp_dict, label_v_dp_dict
        else:
            return label_v_dp_dict


#         return train_image


# Note that the first directory is train(blur), the second directory is label(clear)
def fetch_dataloader(types, label_data_dir,input_t_length,batch_size,downsample,downsample_index,time_interp):
    ########################### Here add a new parameter "blur_data_dir", which is train
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    for split in ['train', 'val', 'use']:
        if split in types:

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(FACESDataset(label_data_dir,input_t_length,train_transformer,is_train=1,downsample_form=downsample,down_sample=downsample_index,time_interp=time_interp), batch_size=batch_size,
                                shuffle=True,
                                )
            elif split=='val':
                dl = DataLoader(FACESDataset(label_data_dir,input_t_length,eval_transformer,is_train=0,downsample_form=downsample,down_sample=downsample_index,time_interp=time_interp), batch_size=batch_size,
                                shuffle=False,
                                )
            else:
                dl = DataLoader(FACESDataset(label_data_dir, input_t_length, eval_transformer, is_train=0,
                                             downsample_form=None,down_sample=downsample_index), batch_size=batch_size,
                                shuffle=False,
                                )

            dataloaders[split] = dl

    return dataloaders