import torch
import os
import numpy as np
dir1='H:\三维流场重建\数据集\训练集\\boundary测试_pth整合\过渡\标签'
dir2='H:\三维流场重建\数据集\训练集\\boundary测试_pth整合\湍流\标签'
dir_new='H:\三维流场重建\数据集\训练集\\boundary测试_pth整合\过渡拼接\标签'
names1=os.listdir(dir1)
names2=os.listdir(dir2)
pathes1=[]
pathes2=[]
for name1,name2 in zip(names1,names2):
    pathes1.append(os.path.join(dir1,name1))
    pathes2.append(os.path.join(dir2, name2))
for path1,path2,name1 in zip(pathes1,pathes2,names1):
    flo_dir1=torch.load(path1)
    flo_dir2=torch.load(path2)
    flo_dir_new={}
    flo_dir_new['v_p']=torch.cat([flo_dir1['v_p'],flo_dir2['v_p']],dim=3).numpy()[:,:,:,8:]
    flo_dir_new['x_coord']=np.concatenate([flo_dir1['x_coord'],flo_dir2['x_coord']],axis=2)[:,:,8:]
    flo_dir_new['y_coord'] = np.concatenate([flo_dir1['y_coord'], flo_dir2['y_coord']], axis=2)[:,:,8:]
    flo_dir_new['z_coord'] = np.concatenate([flo_dir1['z_coord'], flo_dir2['z_coord']], axis=2)[:,:,8:]
    flo_dir_new['t_step']=flo_dir1['t_step']
    flo_dir_new['viscosity'] = flo_dir1['viscosity']
    flo_dir_new['density'] = flo_dir1['density']
    path_new=os.path.join(dir_new,name1)
    torch.save(flo_dir_new,path_new)





