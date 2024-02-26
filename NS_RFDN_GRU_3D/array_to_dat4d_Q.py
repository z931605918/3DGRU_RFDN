
import numpy as np

from scipy.io import loadmat
import os, io, time, re
from glob import glob
from typing import Union, List, Tuple, Optional
import torch
# from src.utils_color import compute_color
from loss_eval_index import cal_Q,load_vel_pth
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
                                                 mode='trilinear', align_corners=False)[0][0].numpy()
        y_coord= torch.nn.functional.interpolate(torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0), size=( d,  h,  w),
                                                 mode='trilinear', align_corners=False)[0][0].numpy()
        z_coord= torch.nn.functional.interpolate(torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0), size=( d,  h,  w),
                                                 mode='trilinear', align_corners=False)[0][0].numpy()
        out_flo['x_coord']=x_coord
        out_flo['y_coord'] =y_coord
        out_flo['z_coord'] =z_coord

        # out_flo['x_coord']=torch.from_numpy(flo['x_coord'])
        # out_flo['y_coord'] = torch.from_numpy(flo['y_coord'])
        # out_flo['z_coord'] = torch.from_numpy(flo['z_coord'])
        out_flo['v_oumiga']=out_V
        out_flos.append(out_flo)

    return pathes,out_flos
def array_to_dat4d(array,save_name,coord,Q):
    array=array[:,:-10,:,80:]
    Q=Q[:-10,:,80:]
    coord_=[]
    for coord__ in coord:
        #coord_.append(coord__[:,:,:-10,:,80:])
        coord_.append(coord__[ :-10, :,80:])
    coord=coord_
    c,z,y,x=array.shape
    with open(os.path.join(save_name), 'w') as txt_in:
        txt_in.write(r'Title = "Flow"' + "\n")  # 写入识别字符行
        txt_in.write(r'VARIABLES = "X", "Y", "Z" ,"U", "V", "W", "Q"' + "\n")
        txt_in.write(r'ZONE T = "Rectangular zone", I = ' + str(int(x)) + r', J = ' + str(int(y)) + r', K = '+str(int(z))+', ZONETYPE=Ordered' + "\n")
        txt_in.write(r'DATAPACKING="POINT"'+"\n")
        # for y in range(0, 100):
        #     for x in range(0, 100):
        #         txt_in.write(str((1. / 99) * x) + " ")
        #         txt_in.write(str((1. / 99) * y) + " ")
        #         txt_in.write(str((the_pred_data[0][0][round(x)][round(y)]).item()) + " ")
        #         txt_in.write(str((the_pred_data[0][1][round(x)][round(y)]).item()) + "\n")
        if coord is None:
            x = np.linspace(0, x - 1, x, dtype=int)
            y = np.linspace(0, y - 1, y, dtype=int)
            z=  np.linspace(0, z - 1, z ,dtype=int)
            Y,Z, X = np.meshgrid(y,z, x)
            in_Y,in_Z,in_x=Y,Z,X

        else:
            x = np.linspace(0, x - 1, x, dtype=int)
            y = np.linspace(0, y - 1, y, dtype=int)
            z=  np.linspace(0, z - 1, z ,dtype=int)
            in_Y,in_Z,in_X = np.meshgrid(y,z, x)
            X,Y,Z=coord
        in_Y=in_Y.reshape(-1,1)
        in_Z = in_Z.reshape(-1, 1)
        in_X = in_X.reshape(-1, 1)
        Y_1 = Y.reshape(-1, 1)
        X_1 = X.reshape(-1, 1)
        Z_1 = Z.reshape(-1,1)

        X_1 = np.round(X_1,5)
        Y_1 = np.round(Y_1, 5)
        Z_1 = np.round(Z_1, 5)

        u = array[0,:,:, :][in_Z,in_Y, in_X]
        v = array[1,:,:, :][in_Z,in_Y, in_X]
        w = array[ 2, :, :, :][in_Z,in_Y, in_X]
        Q=Q[in_Z,in_Y, in_X]

        u = np.round(u,5)
        v = np.round(v, 5)
        w = np.round(w, 5)
        Q = np.round(Q, 5)


        out = np.concatenate([X_1,Y_1,Z_1,u,v,w,Q], axis=1).tolist()
        out_str = ''
        out_str += (str(out).replace('[', ' ').replace(']', '\n').replace(',', ' '))
        txt_in.write(out_str)
        txt_in.close()

if __name__=='__main__':
    path='H:\三维流场重建\数据集\训练集\半球扰流from佘文轩pth\\3DGRU-RFDN'
    save_path='H:\三维流场重建\数据集\训练集\半球扰流from佘文轩pth\dat\\3DGRU-RFDN'
    Q_yu=200
    save=1

    if path.endswith('tricubic'):
        out_pathes, out_flos = load_vel_mat(path)
    else:
        out_pathes, out_flos = load_vel_pth(path)
    names=os.listdir(path)
    save_dirs=[]
    for name in names:
        save_dirs.append(os.path.join(save_path,name[:-4]+'.dat'))
    for flo,save_dir in zip(out_flos[:],save_dirs[:]):
        if 'x_coord' in flo.keys():
            if path.endswith('tricubic'):
                coord=(flo['x_coord'],flo['y_coord'],flo['z_coord'])
                Q = cal_Q(torch.from_numpy(flo['v_oumiga'])[0].unsqueeze(dim=0).unsqueeze(dim=0).cpu().float(),
                          torch.from_numpy(flo['v_oumiga'])[1].unsqueeze(dim=0).unsqueeze(dim=0).cpu().float(),
                          torch.from_numpy(flo['v_oumiga'])[2].unsqueeze(dim=0).unsqueeze(dim=0).cpu().float()
                          ,1,
                          torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0).float(),
                          torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0).float(),
                          torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0).float())
                Q = Q[0][0].numpy()
                Q1 = Q.copy()
                Q1[Q <= Q_yu] = 0
                Q1[Q >= Q_yu] = 1
                print(np.sum(Q1) / (48 * 112 * 280))
                if save:
                    array_to_dat4d(flo['v_oumiga'], save_dir, coord, Q)
            else:
                coord=(flo['x_coord'],flo['y_coord'],flo['z_coord'])
                Q = cal_Q(torch.from_numpy(flo['v_p'])[0].unsqueeze(dim=0).unsqueeze(dim=0).cpu().float(),
                          torch.from_numpy(flo['v_p'])[1].unsqueeze(dim=0).unsqueeze(dim=0).cpu().float(),
                          torch.from_numpy(flo['v_p'])[2].unsqueeze(dim=0).unsqueeze(dim=0).cpu().float()
                          ,1,
                          torch.from_numpy(flo['x_coord']).unsqueeze(dim=0).unsqueeze(dim=0).float(),
                          torch.from_numpy(flo['y_coord']).unsqueeze(dim=0).unsqueeze(dim=0).float(),
                          torch.from_numpy(flo['z_coord']).unsqueeze(dim=0).unsqueeze(dim=0).float())
                Q = Q[0][0].numpy()
                Q1 = Q.copy()
                Q1[Q <= Q_yu] = 0
                Q1[Q >= Q_yu] = 1
                print(np.sum(Q1) / (48 * 112 * 280))
                if save:
                    array_to_dat4d(flo['v_p'], save_dir, coord, Q)
            break
        elif 'x_grid' in flo.keys():
            coord = (flo['x_grid'], flo['y_grid'], flo['z_grid'])
            Q = cal_Q(flo['v_oumiga'][0].unsqueeze(dim=0).unsqueeze(dim=0).cpu(),
                      flo['v_oumiga'][1].unsqueeze(dim=0).unsqueeze(dim=0).cpu()
                      , flo['v_oumiga'][2].unsqueeze(dim=0).unsqueeze(dim=0).cpu()
                      ,1, flo['x_grid'],
                      flo['y_grid'],
                      flo['z_grid'])
            Q = Q[0][0].numpy()
            Q1 = Q.copy()
            Q1[Q <= Q_yu] = 0
            Q1[Q >= Q_yu] = 1
            print(np.sum(Q1) / (48 * 112 * 280))
            if save:
                array_to_dat4d(flo['v_oumiga'].cpu(), save_dir, coord, Q)
            break




    # flo=torch.load('H:\三维流场重建\数据集\训练集\\boundary_测试pth未整合\\transition_bl_x101-612_t1001.pth')
    # save_name='H:\三维流场重建\数据集\训练集\\boundary_测试dat未整合\\transition_bl_x101-612_t1001.dat'
    # coord=(flo['x_coord'],flo['y_coord'],flo['z_coord'])
    # array_to_dat4d(flo['v_p'],save_name,coord)






