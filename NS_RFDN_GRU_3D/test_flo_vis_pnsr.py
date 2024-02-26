import argparse
import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from plot_velocity_fan import read_flow, write_flow
#from model import Generator
from model import *
from srcnn import *
from ssim import ssim
from RFDN import RFDN


import os
def load_flos(folder):
    names=os.listdir(folder)
    flows=[]
    for name in names:
        if name.endswith('flo'):
            path=os.path.join(folder,name)
            flow=read_flow(path).transpose(2,0,1)
            flow=torch.from_numpy(flow).unsqueeze(0).cuda()
            flows.append(flow)
    return flows, names

class Bicubic(torch.nn.Module):
    def __init__(self,
                scale :int
                 ):
        super(Bicubic, self).__init__()
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=True)
    def forward(self, x):
        x = self.upsample(x)
        return x
def calculate_psnr(outputs, labels):
    nume = np.max(outputs, axis=(1, 2, 3), keepdims=True)  # (N,)
    # deno = np.sum((outputs.reshape(-1, 2, 256, 256) - labels.reshape(-1, 2, 256, 256)) ** 2, axis=(1, 2, 3),
    #               keepdims=True)  # (N,)
    deno = np.sum((outputs - labels) ** 2, axis=(1, 2, 3),
                  keepdims=True)

    psnr = 10 * np.sum(np.log((nume * 256) ** 2 / deno) / math.log(10)) / outputs.shape[0]
    return psnr

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--input_folder', default='D:\desktop\YCD\模型s\LR_cylinder',type=str, help='input folder')
parser.add_argument('--label_folder', default='D:\desktop\YCD\模型s\HR_cylinder',type=str, help='label folder')
parser.add_argument('--out_folder', default='D:\desktop\YCD\模型s\输出\RFDN',type=str, help='output folder')
parser.add_argument('--model_name', default='D:\desktop\YCD\模型s\paper里20%混合数据集训练测试权重\RFDN\\RFDN_256batch_1500.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False

input_flows,input_names=load_flos(opt.input_folder)
label_flows,label_names=load_flos(opt.label_folder)
#model = get_paras().cuda()  ##
#model = SRCNN(scale=8).cuda()
model = RFDN(upscale=8).cuda()
#model.cuda().eval()
#summary(model,input_size=(2,64,64),batch_size=2,device='cuda')
bicubic=Bicubic(scale=8).cuda()
#bicubic=Bicubic(scale=8)



model.load_state_dict(torch.load(opt.model_name))

index=1.
total_cost_time=0.
for input,label,input_name,label_name in zip(input_flows,label_flows,input_names,label_names):
    start = time.time()
    input=input.cuda().float()

    out = model(input)

    #out=bicubic(input)
    end=time.time()
    cost_time=end-start
    out_ssim=ssim(out,label).detach().cpu().numpy()
    out=out.detach().cpu().numpy()
    label=label.detach().cpu().numpy()
    out_psnr=calculate_psnr(label,out)
    out=out[0].transpose(1,2,0)
    #write_flow(out,opt.out_folder+'/'+input_name[:-4]+'ssim='+str(out_ssim)+'psnr='+str(out_psnr)+'.flo')
    write_flow(out, opt.out_folder + '/' + input_name)
    index+=1
    total_cost_time+=cost_time

mean_time=total_cost_time/index
print('mean cost %.10f second'%mean_time)
    ##测试程序



















