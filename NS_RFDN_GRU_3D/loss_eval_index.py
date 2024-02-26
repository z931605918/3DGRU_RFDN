import numpy as np
import torch
import os
from scipy.io import loadmat
from N_S_loss_Fan_init_coord import cal_NS_residual_vv_rotate_nonorm,grad_i_x,grad_i,cal_NS_residual_vv_rotate
import matplotlib.pyplot as plt
from ssim import ssim
kernelx = torch.tensor([[[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]],
                        [[-2., 0., 2.],
                         [-4., 0., 4.],
                         [-2., 0., 2.]],
                        [[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]]])
kernelx = kernelx / torch.sum(abs(kernelx))
kernelx = kernelx
kernely = kernelx.transpose(1, 2)
kernelz = kernelx.transpose(2, 0)
kernelx = kernelx.reshape(1, 1, 3, 3, 3)
kernely = kernely.reshape(1, 1, 3, 3, 3)
kernelz = kernelz.reshape(1, 1, 3, 3, 3)

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    max=np.max(img2)
    mean_=np.mean(img2)
    if mse == 0:
        return float('inf')
    else:
        ps=20*np.log10(max/np.sqrt(mse))
        return ps
def normal_rmse_NS(output,label):
    loss_rmse=torch.sqrt(((output[:,0]-label[:,0]).pow(2)+(output[:,1]-label[:,1]).pow(2)+(output[:,2]-label[:,2]).pow(2)).mean()).cpu().numpy()
    #mean=torch.sqrt((label[:,0].pow(2)+label[:,1].pow(2)+label[:,2].pow(2))).cpu().numpy()
    mean = torch.sqrt((label[:, 0].pow(2) + label[:, 1].pow(2) + label[:, 2].pow(2)).mean()).cpu().numpy()

    loss_rmse__ = torch.sqrt(((output[:, 0] - label[:, 0]).pow(2) + (output[:, 1] - label[:, 1]).pow(2) + (
            output[:, 2] - label[:, 2]).pow(2)).mean(axis=1).mean(axis=1).mean(axis=1)).cpu().numpy()
    mean__ = torch.sqrt((label[:, 0].pow(2) + label[:, 1].pow(2) + label[:, 2].pow(2)).mean(axis=1).mean(
        axis=1).mean(axis=1)).cpu().numpy()
    return loss_rmse/mean
def normal_rmse_NS_iso(output,label):
    loss_rmse=torch.sqrt(((output[:,0]-label[:,0]).pow(2)+(output[:,1]-label[:,1]).pow(2)+(output[:,2]-label[:,2]).pow(2)).mean()).cpu().numpy()
    #mean=torch.sqrt((label[:,0].pow(2)+label[:,1].pow(2)+label[:,2].pow(2))).cpu().numpy()
    mean = torch.sqrt((label[:, 0].pow(2) + label[:, 1].pow(2) + label[:, 2].pow(2)).mean()).cpu().numpy()
    loss_rmse__ = torch.sqrt(((output[:, 0] - label[:, 0]).pow(2) + (output[:, 1] - label[:, 1]).pow(2) + (
            output[:, 2] - label[:, 2]).pow(2)).mean(axis=1).mean(axis=1).mean(axis=1)).cpu().numpy()
    mean__ = torch.sqrt((label[:, 0].pow(2) + label[:, 1].pow(2) + label[:, 2].pow(2)).mean(axis=1).mean(
        axis=1).mean(axis=1)).cpu().numpy()
    mean__2= torch.sqrt((output[:, 0].pow(2) + output[:, 1].pow(2) + output[:, 2].pow(2)).mean(axis=1).mean(
        axis=1).mean(axis=1)).cpu().numpy()
    return loss_rmse
def normal_rmse_cont(output,label):
    loss_rmse=torch.sqrt(((output[:,0]-label[:,0]).pow(2).mean()))
    mean=torch.sqrt(label.pow(2).mean()).cpu()*3
    return loss_rmse/mean
def normal_rmse_cont_iso(output,label):
    loss_rmse=torch.sqrt(((output[:,0]-label[:,0]).pow(2).mean()))
    mean=torch.sqrt(label.pow(2).mean()).cpu()*3
    return loss_rmse

def normal_rmse(output,label):
    loss_rmse__ = torch.sqrt(((output[:, 0] - label[:, 0]).pow(2) + (output[:, 1] - label[:, 1]).pow(2) + (
            output[:, 2] - label[:, 2]).pow(2)).mean(axis=1).mean(axis=1).mean(axis=1)).cpu().numpy()
    mean__ = torch.sqrt((label[:, 0].pow(2) + label[:, 1].pow(2) + label[:, 2].pow(2)).mean(axis=1).mean(
        axis=1).mean(axis=1)).cpu().numpy()
    mean__2 = torch.sqrt((output[:, 0].pow(2) + output[:, 1].pow(2) + output[:, 2].pow(2)).mean(axis=1).mean(
        axis=1).mean(axis=1)).cpu().numpy()

    loss_rmse=torch.sqrt(((output[:,0]-label[:,0]).pow(2)+(output[:,1]-label[:,1]).pow(2)+(output[:,2]-label[:,2]).pow(2)).mean()).cpu().numpy()
    mean=torch.sqrt((label[:,0].pow(2)+label[:,1].pow(2)+label[:,2].pow(2)).mean()).cpu().numpy()
    return loss_rmse/mean
def cal_space_grad(u,pad_num,x_step,y_step,z_step):
    x_diff_grid=abs(grad_i_x(x_step.float(),kernelx,1,pad=pad_num))
    y_diff_grid=abs(grad_i_x(y_step.float(),kernely,1,pad=pad_num))
    z_diff_grid=abs(grad_i_x(z_step.float(),kernelz,1,pad=pad_num))
    u_x=  grad_i_x(u,kernelx, 1,pad=pad_num)/x_diff_grid
    u_y=  grad_i_x(u,kernely, 1,pad=pad_num)/y_diff_grid
    u_z = grad_i_x(u,kernelz, 1,pad=pad_num)/z_diff_grid
    return u_x,u_y,u_z
def cal_oumiga(u,v,w,pad_num,x_step,y_step,z_step):
    u_x, u_y, u_z=  cal_space_grad(u,pad_num,x_step,y_step,z_step)
    v_x, v_y, v_z = cal_space_grad(v, pad_num, x_step, y_step, z_step)
    w_x, w_y, w_z = cal_space_grad(w, pad_num, x_step, y_step, z_step)
    oumiga_x=(w_y-v_z)*0.5
    oumiga_y=(u_z-w_x)*0.5
    oumiga_z=(v_x-u_y)*0.5
    return oumiga_x,oumiga_y,oumiga_z


def cal_Q(u,v,w,pad_num,x_step,y_step,z_step):
    u_x,u_y,u_z=cal_space_grad(u,pad_num,x_step,y_step,z_step)
    v_x, v_y, v_z = cal_space_grad(v, pad_num, x_step, y_step, z_step)
    w_x, w_y, w_z = cal_space_grad(w, pad_num, x_step, y_step, z_step)
    Q=(u_x*v_y-v_x*u_y)+(v_y*w_z-w_y*v_z)+(u_x*w_z-w_x*u_z)
    return Q
def load_vel_pth(dir):
    names=os.listdir(dir)
    pathes=[]
    flos=[]
    for name in names:
        pathes.append(os.path.join(dir,name))
        flos.append(torch.load(os.path.join(dir,name)))
    return pathes,flos
def load_vel_mat(dir):
    names=os.listdir(dir)
    pathes=[]
    flos=[]
    for name in names:
        pathes.append(os.path.join(dir,name))
        flos.append(loadmat(os.path.join(dir,name)))
    out_flos=[]
    for flo in flos:
        out_flo={}
        U=flo['v_interp'][:112]
        V=flo['v_interp'][112:224]
        W=flo['v_interp'][224:]
        U=U.transpose(2,0,1)
        V=V.transpose(2,0,1)
        W=W.transpose(2,0,1)
        out_V=np.stack([U,V,W],axis=0)
        out_flo['t_step']=torch.from_numpy(flo['t_step'][0])
        out_flo['density']=torch.from_numpy(flo['density'][0])
        out_flo['viscosity']=torch.from_numpy(flo['viscosity'][0])
        c,d,h,w=out_V.shape
        x_coord= torch.nn.functional.interpolate(torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0), size=( d,  h,  w),
                                                 mode='trilinear', align_corners=False)[0][0]
        y_coord= torch.nn.functional.interpolate(torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0), size=( d,  h,  w),
                                                 mode='trilinear', align_corners=False)[0][0]
        z_coord= torch.nn.functional.interpolate(torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0), size=( d,  h,  w),
                                                 mode='trilinear', align_corners=False)[0][0]
        out_flo['x_coord']= x_coord
        out_flo['y_coord'] =y_coord
        out_flo['z_coord'] =z_coord

        # out_flo['x_coord']=torch.from_numpy(flo['x_coord'])
        # out_flo['y_coord'] = torch.from_numpy(flo['y_coord'])
        # out_flo['z_coord'] = torch.from_numpy(flo['z_coord'])
        out_flo['v_oumiga']=torch.from_numpy(out_V)
        out_flos.append(out_flo)

    return pathes,out_flos
def load_out_label(out_dir,label_dir):

    if out_dir.endswith('linear') or out_dir.endswith('cubic'):
        out_pathes,out_flos=load_vel_mat(out_dir)
    else :
        out_pathes,out_flos=load_vel_pth(out_dir)
    label_pathes,label_flos=load_vel_pth(label_dir)
    return out_pathes,out_flos,label_pathes,label_flos

def cal_L2_vel(out_dir: str, label_dir: str):
    out_pathes, out_flos, label_pathes, label_flos = load_out_label(out_dir, label_dir)
    out_vs = []
    label_vs = []
    for out_flo, label_flo in zip(out_flos, label_flos):
        pad_num = 1
        out_vel = out_flo['v_oumiga'][:3]
        out_u=out_vel[0].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:,:,:-10,:,80:]
        out_v=out_vel[1].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:,:,:-10,:,80:]
        out_w=out_vel[2].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:,:,:-10,:,80:]
        out_vel = torch.cat([out_u, out_v, out_w], dim=1)
        out_vs.append(out_vel)

        label_vel=torch.from_numpy(label_flo['v_p'][:3]).cpu()
        label_u=label_vel[0].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:,:,:-5,:,40:]
        label_v=label_vel[1].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:,:,:-5,:,40:]
        label_w=label_vel[2].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:,:,:-5,:,40:]
        label_vel=torch.cat([label_u,label_v,label_w],dim=1)
        label_vs.append(label_vel)

        label_oumiga_np=label_vel.numpy()
        out_oumiga_np=out_vel.numpy()

        # ampls_label=np.log10(abs(torch.fft.fftn(label_vel)/label_oumiga_np.size)**2).numpy()
        # ampls_out=np.log10(abs(torch.fft.fftn(out_vel)/out_oumiga_np.size)**2).numpy()



    out_vs=torch.cat(out_vs,dim=0)
    label_vs=torch.cat(label_vs,dim=0)

    mean_out_vs=torch.mean(out_vs,dim=0)
    mean_label_vs=torch.mean(label_vs,dim=0)


    # y=torch.log10(torch.linspace(0,79,80).int()/80*1000)
    # u_out=mean_out_vs[0,32,:,64]
    # u_label=mean_label_vs[0,32,:,64]
    # plt.plot(y,u_label)
    # plt.plot(y,u_out)
    # plt.show()
    j=0
    i=10
    plt.subplot(211)
    plt.imshow(label_vs[j, 0, 4,:].cpu().numpy(), cmap='jet',vmax=1.5,vmin=-0.5)
    plt.subplot(212)
    plt.imshow(out_vs[j, 0, 9,:].cpu().numpy(), cmap='jet',vmax=1.5,vmin=-0.5)

    # plt.subplot(313)
    # plt.imshow((out_vs[j, 0, i,:].cpu().numpy() - label_vs[j, 0, i,:].cpu().numpy()),
    #            cmap='seismic',vmax=0.15,vmin=-0.15)
    plt.colorbar(orientation='horizontal')
    plt.show()
    L2_v=normal_rmse(out_vs,label_vs)
    print('L2 v =%.5f'%L2_v)
    L2_v=psnr(out_vs.numpy(),label_vs.numpy())
    print('psnr v =%.5f'%L2_v)
    L2_v=ssim(out_vs[:5].double(),label_vs[:5].double())
    print('ssim v =%.5f'%L2_v)






    return L2_v

def plot_line(out_dir: str, label_dir: str):
    out_pathes, out_flos, label_pathes, label_flos = load_out_label(out_dir, label_dir)
    out_vs = []
    label_vs = []
    for out_flo, label_flo in zip(out_flos, label_flos):
        pad_num = 1
        out_vel = out_flo['v_oumiga'][:3]

        out_u = out_vel[0].unsqueeze(dim=0).unsqueeze(dim=0).cpu()
        out_v = out_vel[1].unsqueeze(dim=0).unsqueeze(dim=0).cpu()
        out_w = out_vel[2].unsqueeze(dim=0).unsqueeze(dim=0).cpu()
        out_vel = torch.cat([out_u, out_v, out_w], dim=1)
        out_vs.append(out_vel)

        label_vel = torch.from_numpy(label_flo['v_p'])[:3].cpu()
        label_u = label_vel[0].cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        label_v = label_vel[1].cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        label_w = label_vel[2].cpu().unsqueeze(dim=0).unsqueeze(dim=0)
        label_vel = torch.cat([label_u, label_v, label_w], dim=1)
        label_vs.append(label_vel)

        label_oumiga_np = label_vel.numpy()
        out_oumiga_np = out_vel.numpy()

        # ampls_label=np.log10(abs(torch.fft.fftn(label_vel)/label_oumiga_np.size)**2).numpy()
        # ampls_out=np.log10(abs(torch.fft.fftn(out_vel)/out_oumiga_np.size)**2).numpy()

    out_vs = torch.cat(out_vs, dim=0)
    label_vs = torch.cat(label_vs, dim=0)

    mean_out_vs = torch.mean(out_vs, dim=0)
    mean_label_vs = torch.mean(label_vs, dim=0)




    u_out=mean_out_vs[0,32,:,64]
    u_label=mean_label_vs[0,32,:,64]
    return u_label,u_out




def cal_L2_oumiga(out_dir:str,label_dir:str):
    out_pathes, out_flos, label_pathes, label_flos=load_out_label(out_dir,label_dir)
    out_oumigas=[]
    label_oumigas=[]
    for out_flo, label_flo in zip(out_flos,label_flos):
        pad_num=0

        label_vel=torch.from_numpy(label_flo['v_p'][:3]).cpu().float()
        label_x_grid=  torch.from_numpy(label_flo['x_coord']).cpu().unsqueeze(dim=0).unsqueeze(dim=0)[ :,:,:-5,:,40:].float()
        label_y_grid = torch.from_numpy(label_flo['y_coord']).cpu().unsqueeze(dim=0).unsqueeze(dim=0)[ :,:,:-5,:,40:].float()
        label_z_grid = torch.from_numpy(label_flo['z_coord']).cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, :,:-5,:,40:].float()
        label_u=label_vel[0].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[ 0:1, :,:-5,:,40:]
        label_v=label_vel[1].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[ 0:1, :,:-5,:,40:]
        label_w=label_vel[2].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[ 0:1, :,:-5,:,40:]
        label_o_x,label_o_y,label_o_z=cal_oumiga(label_u,label_v,label_w,pad_num,label_x_grid,label_y_grid,label_z_grid)
        label_oumiga = torch.cat([label_o_x, label_o_y, label_o_z], dim=1)
        label_oumigas.append(label_oumiga)

        out_vel = out_flo['v_oumiga'][:3].float()
        out_x_grid=  out_flo['x_grid'].cpu()[:, :,:-10,:,80:].float()
        out_y_grid = out_flo['y_grid'].cpu()[:, :,:-10,:,80:].float()
        out_z_grid = out_flo['z_grid'].cpu()[:, :,:-10,:,80:].float()
        out_u=out_vel[0].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:, :,:-10,:,80:]
        out_v=out_vel[1].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:, :,:-10,:,80:]
        out_w=out_vel[2].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:, :,:-10,:,80:]
        out_o_x,out_o_y,out_o_z=cal_oumiga(out_u,out_v,out_w,pad_num,out_x_grid,out_y_grid,out_z_grid)
        out_oumiga=torch.cat([out_o_x,out_o_y,out_o_z],dim=1)
        out_oumigas.append(out_oumiga)

        label_oumiga_np=label_oumiga.numpy()
        out_oumiga_np=out_oumiga.numpy()

       #y=0.01227*网格
    out_oumigas=torch.cat(out_oumigas,dim=0)
    label_oumigas=torch.cat(label_oumigas,dim=0)
    i=10
    plt.subplot(211)
    plt.imshow(label_oumigas[0,2,4,:], cmap='jet', vmax=15, vmin=-15)
    plt.subplot(212)
    plt.imshow(out_oumigas[0,2, 9,:], cmap='jet', vmax=15, vmin=-15)
    plt.colorbar(orientation='horizontal')
    # plt.subplot(313)
    # plt.imshow((out_oumigas[1, 1,4,:] - label_oumigas[1, 1,:,5]), cmap='seismic', vmax=3, vmin=-3)
    # plt.colorbar(orientation='horizontal')
    plt.show()


    l2_oumiga=normal_rmse(out_oumigas,label_oumigas)
    psnr_oumiga=psnr(out_oumigas.numpy(),label_oumigas.numpy())
    print('vorticiy_loss=%.4f' % l2_oumiga)
    print('vorticiy_psnr=%.4f'%psnr_oumiga)
    # ssim_oumiga=ssim(out_oumigas,label_oumigas)
    # print('vorticiy_ssim=%.4f'%ssim_oumiga)

    return l2_oumiga

def cal_L2_NS(out_dir:str,label_dir:str):
    out_pathes, out_flos, label_pathes, label_flos=load_out_label(out_dir,label_dir)
    out_vs = []
    label_vs = []
    for out_flo, label_flo in zip(out_flos,label_flos):
        pad_num=1
        label_vel=torch.from_numpy(label_flo['v_p'][:3]).cpu()
        label_x_grid=torch.from_numpy(label_flo['x_coord']).cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, 0:1, :, 10:, :]
        label_y_grid = torch.from_numpy(label_flo['y_coord']).cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, 0:1, :, 10:, :]
        label_z_grid = torch.from_numpy(label_flo['z_coord']).cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, 0:1, :, 10:, :]
        label_u=label_vel[0].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, 0:1, :, 10:, :]
        label_v=label_vel[1].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, 0:1, :, 10:, :]
        label_w=label_vel[2].cpu().unsqueeze(dim=0).unsqueeze(dim=0)[:, 0:1, :, 10:, :]
        label_t=label_flo['t_step']
        label_vis = label_flo['viscosity']
        out_vel = out_flo['v_oumiga'][:3]
        out_u=out_vel[0].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:, 0:1, :, 10:, :]
        out_v=out_vel[1].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:, 0:1, :, 10:, :]
        out_w=out_vel[2].unsqueeze(dim=0).unsqueeze(dim=0).cpu()[:, 0:1, :, 10:, :]
        out_vel = torch.cat([out_u, out_v, out_w], dim=1)
        out_vs.append(out_vel)
        label_vel=torch.cat([label_u,label_v,label_w],dim=1)
        label_vs.append(label_vel)

    out_vs=torch.cat(out_vs,dim=0)
    label_vs=torch.cat(label_vs,dim=0)
    group_num=int(out_vs.shape[0]/5)
    label_NSs=[]
    out_NSs=[]
    for i in range(group_num):
        label_aresx,label_aresy,label_aresz,label_cont=cal_NS_residual_vv_rotate(label_vs[i*5:i*5+5,0:1].cuda(), label_vs[i*5:i*5+5,1:2].cuda(), label_vs[i*5:i*5+5,2:3].cuda(),
                                  label_x_grid.cuda(), label_y_grid.cuda(), label_z_grid.cuda(), label_t, density=1, viscosity=label_vis)
        out_aresx,out_aresy,out_aresz,out_cont=cal_NS_residual_vv_rotate(out_vs[i*5:i*5+5,0:1].cuda(), out_vs[i*5:i*5+5,1:2].cuda(), out_vs[i*5:i*5+5,2:3].cuda(),
                                  label_x_grid.cuda(), label_y_grid.cuda(), label_z_grid.cuda(), label_t, density=1, viscosity=label_vis)
        label_NS=torch.cat([label_aresx.cpu(),label_aresy.cpu(),label_aresz.cpu()],dim=1)
        out_NS=torch.cat([out_aresx.cpu(),out_aresy.cpu(),out_aresz.cpu()],dim=1)
        label_NSs.append(label_NS)
        out_NSs.append(out_NS)
    label_NS = torch.cat(label_NSs, dim=0)
    out_NS = torch.cat(out_NSs, dim=0)


    # i=10
    # plt.subplot(131)
    # plt.imshow(label_aresy[0, 0,:, i, :].cpu().numpy(), cmap='jet', vmax=1, vmin=-1)
    # plt.colorbar(shrink=0.5)
    # plt.subplot(132)
    # plt.imshow(out_aresy[0, 0,:, i, :].cpu().numpy(), cmap='jet', vmax=1, vmin=-1)
    # plt.colorbar(shrink=0.5)
    # plt.subplot(133)
    # plt.imshow((out_aresy[0, 0,:, i, :].cpu().numpy() - label_aresy[0, 0,:, i, :].cpu().numpy()) ,
    #            cmap='seismic', vmax=2, vmin=-2)
    # plt.colorbar(shrink=0.5)
    # plt.show()

    L2_NS=normal_rmse_NS_iso(out_NS,label_NS)
    # label_NSMean=label_NSMean.mean(axis=1).mean(axis=1).mean(axis=1)
    # L2_NS_mean=L2_NS.mean(axis=1).mean(axis=1).mean(axis=1)
    L2_cont=normal_rmse_cont_iso(out_cont,label_cont)
    NS_psnr=psnr(out_NS.cpu().numpy(),label_NS.cpu().numpy())
#    NS_ssim=ssim(out_NS,label_NS)
    print('ns psnr=%.4f' % NS_psnr)
   # print('ns ssim=%.4f' % NS_ssim)
    print('L2 ns=%.4f'%L2_NS)
    print('L2 cont=%.4f'%L2_cont)
    return L2_NS,L2_cont

def qiepian_plot():
    out_path1= 'H:\三维流场重建\数据集\训练集\channel测试集_bigpth\middle\\tricubic'
    out_path2 = 'H:\三维流场重建\数据集\训练集\channel测试集_bigpth\middle\\Bay-3DGRU-RFDN-NSATT'
    out_path3 = 'H:\三维流场重建\数据集\训练集\channel测试集_bigpth\middle\\Bay-3DGRU-RFDN'
    out_path4 = 'H:\三维流场重建\数据集\训练集\channel测试集_bigpth\middle\\3DGRU-RFDN-NSATT'
    out_path5 = 'H:\三维流场重建\数据集\训练集\channel测试集_bigpth\middle\\3DGRU-RFDN'
    label_path='H:\三维流场重建\数据集\训练集\channel测试集_bigpth\middle\标签'

    u_label,out_1=plot_line(out_path1,label_path)
    u_label, out_2 = plot_line(out_path2, label_path)
    u_label, out_3 = plot_line(out_path3, label_path)
    u_label, out_4 = plot_line(out_path4, label_path)
    u_label, out_5 = plot_line(out_path5, label_path)
    y = torch.linspace(1, 80, 80).int() / 80
    u_fr=0.0499
    vis=5e-5
    y_fr=u_fr*y/vis
    u_=2.44*np.log10(y_fr)+5.2
    color = ['black', 'blue', 'red', 'orange',  'yellow','yellowgreen']
    line = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed']

    plt.semilogx(y_fr, u_label/u_fr,color=color[0],linestyle=line[0],label='DNS')
    #plt.semilogx(y_fr, out_1/u_fr,color=color[1],linestyle=line[1],label='2Tricubic')
    plt.semilogx(y_fr, out_2/u_fr,color=color[2],linestyle=line[2],label='Bay-3DGRU-RFDN-NSATT')
    plt.semilogx(y_fr, out_3/u_fr,color=color[3],linestyle=line[3],label='Bay-3DGRU-RFDN')
    plt.semilogx(y_fr, out_4/u_fr,color=color[4],linestyle=line[4],label='3DGRU-RFDN-NSATT')
    plt.semilogx(y_fr, out_5/u_fr,color=color[5],linestyle=line[5],label='3DGRU-RFDN')
    #plt.semilogx(y_fr, u_, color=color[0], linestyle=line[1], label='3DGRU-RFDN')
    plt.yticks(fontproperties = 'Times New Roman', size = 25)
    plt.xticks(fontproperties = 'Times New Roman', size = 25)
    plt.xlabel('y+',fontsize=25,fontproperties = 'Times New Roman')
    plt.ylabel('u+',fontsize=25,fontproperties = 'Times New Roman')
    plt.legend(loc='upper left', prop={'size': 15})

    plt.show()



if __name__=='__main__':
    out_path='H:\三维流场重建\数据集\训练集\半球扰流from佘文轩pth\\3DGRU-RFDN'
    label_path='H:\三维流场重建\数据集\训练集\半球扰流from佘文轩pth\标签part'
    #L2_ves=cal_L2_vel(out_path,label_path)
    L2_oumiga=cal_L2_oumiga(out_path,label_path)
    # L2_NS,L2_cont = cal_L2_NS(out_path, label_path)




