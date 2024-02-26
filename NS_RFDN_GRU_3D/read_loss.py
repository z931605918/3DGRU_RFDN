import os
import torch
names=os.listdir('D:\desktop\\3DFlowfield_reconstruction\\NS_RFDN_GRU_3D\LOSS')
pathes=[]
i=0
for name in names:
    path=os.path.join('D:\desktop\\3DFlowfield_reconstruction\\NS_RFDN_GRU_3D\LOSS',name)
    pathes.append(path)
    loss=torch.load(path)
    print(i)
    print(loss)
    i+=1

