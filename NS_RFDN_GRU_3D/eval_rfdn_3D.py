"""Train the model"""

import argparse
import logging
import os
import math
import numpy as np
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from BayesNN import BayesNN
from SVGD import SVGD
import utils
from torch.nn.utils import clip_grad_norm
from torch.optim import lr_scheduler
from dataloder import *
import torchvision.transforms.functional as F
#from skimage import io
import scipy.interpolate as interp
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from model_RFDN.RFDN_3D_downsample_GRU_input_divide import RFDN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ssim import ssim

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./parameters', help="Directory containing params.json")
parser.add_argument('--trained_para',default='D:\desktop\\3DFlowfield_reconstruction\\NS_RFDN_GRU_3D\
\\parameters_divide_not_down_贝叶斯+ATT_notime\\finetune_0785.pth',help='预训练参数')
#
parser.add_argument('--use_output_savepath',
                    default='H:\三维流场重建\数据集\训练集\半球扰流from佘文轩pth\Bay-3DGRU-RFDN-NSATT',
                    help="使用时输出保存地址")
parser.add_argument('--eval_label_data_dir',
                    default='H:\三维流场重建\数据集\训练集\iso_测试集big_pth\corner\标签', help="测试集")
parser.add_argument('--NS_ECA',default=1,help='使用NS方程计算注意力')
parser.add_argument('--test_or_eval_or_use',default='eval',help='测试，评估或使用')
parser.add_argument('--input_t_length', default=5, help="每组流场数量")
parser.add_argument('--upsample_index',default=2,help='空间插值上采样倍率')
parser.add_argument('--time_interp',default=0,help='时间插值上采样倍率')
parser.add_argument('--interp_form',default='cubic',help='插值形式,linear , cubic, spline')
parser.add_argument('--downsample_form',default='mean',help='降采样形式mean/gauss')
parser.add_argument('--start_epoch',default=500,help='开始迭代的位数')
parser.add_argument('--loss_save_path', default='D:\desktop\\3DFlowfield_reconstruction\\NS_RFDN_GRU_3D\LOSS',
                    help="误差保存地址")
parser.add_argument('--optim', default='adam')
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
parser.add_argument('--use_bayes',default=1,help='是否使用贝叶斯估计')
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--batch_size', type=int, default=1 ,help='设置为1')
parser.add_argument('--start_iter_num', default=1,help='设置为1 勿动')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lrf', type=float, default=0.001)
parser.add_argument('--var_eq',default=1e-4,help='贝叶斯置信度')
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()
#net = get_paras().to(device)
# python -m visdom.server
global_loss = []
def grid_upsample(target_dict):
    x_coord_line = target_dict['x_grid'][0][0, 0, 0, 0, :]
    x_coord_min = x_coord_line[0]
    x_coord_max = x_coord_line[-1]
    x_coord_step = x_coord_line[1] - x_coord_line[0]
    x_coord_line_new = torch.linspace(x_coord_min - 0.5 * x_coord_step, x_coord_max + 0.5 * x_coord_step,
                                      x_coord_line.shape[0] * 2)

    y_coord_line = target_dict['y_grid'][0][0, 0, 0, :, 0]
    y_coord_min = y_coord_line[0]
    y_coord_max = y_coord_line[-1]
    y_coord_step = y_coord_line[1] - y_coord_line[0]
    y_coord_line_new = torch.linspace(y_coord_min - 0.5 * y_coord_step, y_coord_max + 0.5 * y_coord_step,
                                      y_coord_line.shape[0] * 2)

    z_coord_line = target_dict['z_grid'][0][0, 0, :, 0, 0]
    z_coord_min = z_coord_line[0]
    z_coord_max = z_coord_line[-1]
    z_coord_step = z_coord_line[1] - z_coord_line[0]
    z_coord_line_new = torch.linspace(z_coord_min - 0.5 * z_coord_step, z_coord_max + 0.5 * z_coord_step,
                                      z_coord_line.shape[0] * 2)

    z_grid, y_grid, x_grid = torch.meshgrid(z_coord_line_new, y_coord_line_new, x_coord_line_new)
    return x_grid,y_grid,z_grid
# if opt.matlab_interp3:
#     eng = matlab.engine.start_matlab()
def train(model, optimizer, loss_fn, dataloader, metrics, epoch):
    # set model to training mode
    model.train()
    # print('this the type of the model', type(model))

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    current_learning_rate = optimizer.defaults['lr']

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_dict, labels_dict) in enumerate(dataloader):
            # move to GPU if available
            train_batch=train_dict['v_dp']
            labels_batch=labels_dict['v_dp']
            train_batch, labels_batch = train_batch.to(device)[0], labels_batch.to(device)[0]
            # convert to torch Variables
            # print(train_batch.shape, labels_batch.shape)
            # assert 1 > 9
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)[args.start_iter_num:]
            # compute model output and loss
            torch.set_default_dtype(torch.float64)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            #             clip_grad_norm(model.parameters(), 0.01 / current_learning_rate)
            # performs updates using calculated gradients
            optimizer.step()
            # Evaluate summaries only once in a while
            if i % 20 == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            global_loss.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)
def evaluate(bayes_nn, dataloader,eval_loss_savefilename):
    # set model to evaluation mode
    with torch.no_grad():
        bayes_nn.eval().cuda()
        # summary for current eval loop
        summ = []
        total_outputs=[]
        total_labels=[]
        count = 0
        # compute metrics over the dataset
        total_loss = 0.
        total_loss_psnr=0.
        total_loss_nsx=0.
        total_loss_nsy = 0.
        total_loss_nsz = 0.
        total_loss_nscont = 0.
        total_loss_ssim=0.
        #viz_1 = visdom.Visdom()
        for i, (input_dict, target_dict) in enumerate(dataloader):
            # move to GPU if available
            loss_nsx=0.
            loss_nsy = 0.
            loss_nsz = 0.
            loss_nscont = 0.
            loss_m_show = 0.
            steps = (
            target_dict['x_grid'][0].to(device),
            target_dict['y_grid'][0].to(device),
            target_dict['z_grid'][0].to(device))
            kernel = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
            steps2=(
            kernel(target_dict['x_grid'][0].to(device)),
            kernel(
                target_dict['y_grid'][0].to(device)),
            kernel(
            target_dict['z_grid'][0].to(device))
            )
            density = target_dict['density'][0]
            viscosity = target_dict['viscosity'][0]
            t_step = target_dict['t_step'][0]
            input = input_dict['v_oumiga'].to(device)[0]
            target = target_dict['v_oumiga'].to(device)[0][:, :3]
            output_is = []
            # compute model output
            output =0
            _,_,Ld,Lh,Lw=input.shape
            if args.upsample_index == 2:
                for i in range(len(model)):
                    output_i = bayes_nn[i].forward(input, steps, t_step, density, viscosity)
                    output+=output_i[:,:3]
                output/=3

            if args.upsample_index == 4:
                for i in range(len(model)):
                    output_i = bayes_nn[i].forward(input, steps2, t_step, density, viscosity)
                    output+=output_i
                output /= 3

                output_ = 0
                for i in range(len(model)):
                    output_i = bayes_nn[i].forward(output, steps, t_step, density, viscosity, t_interp=False)
                    output_ += output_i
                output_ /= 3
                output = output_[:,:3]
                # output_i=torch.nn.functional.interpolate(input, (2*Ld, 2*Lh, 2*Lw), mode='trilinear', align_corners=False)
                # output_0=output_i[0]
                # output_1 = output_i[1]
                # output_2 = output_i[2]
                # output_05=0.5*(output_0+output_1)
                # output_15 = 0.5 * (output_1 + output_2)
                # output_i=torch.stack([output_0,output_05,output_1,output_15,output_2],dim=0)


            log_eq, loss_nsx, loss_nsy, loss_nsz, loss_nscont, res_x_out, res_x_tag = bayes_nn.criterion_grid(output,
                                                                                                              target,
                                                                                                              steps,
                                                                                                              t_step,
                                                                                                              density,
                                                                                                              viscosity,
                                                                                                              i, 1)


            output_np=output.cpu().numpy()
            label_np=target.cpu().numpy()
            loss_rmse,mean = normal_rmse(output,target)
            loss_rmse=loss_rmse/mean
            loss_psnr=psnr(output[:, :3].cpu().numpy(), target[:, :3].cpu().numpy())
            loss_ssim=ssim(output[:, :3].cpu().numpy(),target[:, :3].cpu().numpy())
            total_outputs.append(output)
            total_labels.append(target)

            #
            # target_np=target.detach().cpu().numpy()
            #
            target_fft_cpu = torch.log10(abs(torch.fft.fftn(target[:, :3].cpu(), norm='forward')))
            output_fft_cpu = torch.log10(abs(torch.fft.fftn(output[:, :3].cpu(), norm='forward')))
            loss_rmse_fft, mean_fft = normal_rmse(output_fft_cpu, target_fft_cpu)
            loss_rmse_fft_norm=loss_rmse_fft/mean_fft

            # target_fft=target_fft.detach().cpu().numpy()
            # target_fft_cpu=target_fft_cpu.detach().cpu().numpy()
            # fft_max=float(target_fft.max())
            # fft_min=float(target_fft.min())
            # plt.subplot(231)
            # plt.imshow(targetfft_np[ 0, 0, ], cmap='jet')
            # plt.colorbar(orientation='horizontal')
            # plt.subplot(232)
            # plt.imshow(target_fft[ 0, 0, ], cmap='jet')
            # plt.colorbar(orientation='horizontal')
            #
            # plt.subplot(233)
            # plt.imshow(target_fft_cpu[ 0, 0, ], cmap='jet')
            # plt.colorbar(orientation='horizontal')
            #
            # plt.subplot(235)
            # plt.imshow(target[ 0,  0].cpu().detach().numpy(), cmap='jet')
            # plt.colorbar(orientation='horizontal')
            # plt.show()

            print('rmse_loss is %.5f , psnr is %.5f  ssim is %.5f fft is %.5f'%(loss_rmse,loss_psnr,loss_ssim, loss_rmse_fft_norm))
            loss_nsx=  loss_nsx /len(model)
            loss_nsy = loss_nsy / len(model)
            loss_nsz = loss_nsz / len(model)
            loss_nscont=loss_nscont / len(model)
            total_loss_psnr+=loss_psnr
            total_loss+=loss_rmse
            total_loss_nsx+=loss_nsx.cpu().numpy()
            total_loss_nsy+=loss_nsy.cpu().numpy()
            total_loss_nsz+=loss_nsz.cpu().numpy()
            total_loss_nscont+=loss_nscont.cpu().numpy()
            total_loss_ssim+=loss_ssim
        _,_,d,h,w=target.shape
        # viz_1.heatmap(X=target[-2, 0, 20].cpu().numpy(), win='ulabel',
        #               opts=dict(title='ulabel', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=output[-2, 0, 20].cpu().numpy(), win='uout',
        #               opts=dict(title='uout1', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=target[-2, 1, 20].cpu().numpy(), win='vlabel',
        #               opts=dict(title='vlabel', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=output[-2, 1, 20].cpu().numpy(), win='vout',
        #               opts=dict(title='vout1', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=target[-2, 2, 20].cpu().numpy(), win='wlabel',
        #               opts=dict(title='wlabel', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=output[-2, 2, 20].cpu().numpy(), win='wout',
        #               opts=dict(title='wout1', colormap='rainbow', xmax=1, xmin=-1))
        # input_show = nn.functional.interpolate(input, size=(d, h, w), mode='trilinear', align_corners=False)
        # viz_1.heatmap(X=input_show[-1, 0, 20].cpu().numpy(), win='uinput',
        #               opts=dict(title='uinput', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=input_show[-1, 1, 20].cpu().numpy(), win='vinput',
        #               opts=dict(title='vinput', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=input_show[-1, 2, 20].cpu().numpy(), win='winput',
        #               opts=dict(title='winput', colormap='rainbow', xmax=1, xmin=-1))
        #
        # viz_1.heatmap(X=input_show[-1, 0, 20].cpu().numpy() - target[-1, 0, 20].cpu().numpy(), win='uinput-res',
        #               opts=dict(title='uinput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
        # viz_1.heatmap(X=input_show[-1, 1, 20].cpu().numpy() - target[-1, 1, 20].cpu().numpy(), win='vinput-res',
        #               opts=dict(title='vinput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
        # viz_1.heatmap(X=input_show[-1, 2, 20].cpu().numpy() - target[-1, 2, 20].cpu().numpy(), win='winput-res',
        #               opts=dict(title='winput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
        #
        # viz_1.heatmap(X=target[-2, 0, 20].cpu().numpy() - output[-2, 0, 20].cpu().numpy(), win='ures',
        #               opts=dict(title='u残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
        # viz_1.heatmap(X=target[-2, 1, 20].cpu().numpy() - output[-2, 1, 20].cpu().numpy(), win='vres',
        #               opts=dict(title='v残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
        # viz_1.heatmap(X=target[-2, 2, 20].cpu().numpy() - output[-2, 2, 20].cpu().numpy(), win='wres',
        #               opts=dict(title='w残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
        # viz_1.heatmap(X=loss_1i[-2, 0, 20].detach().cpu().numpy(), win='xres',
        #               opts=dict(title='Ns_x残差相减', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=loss_2i[-2, 0, 20].detach().cpu().numpy(), win='yres',
        #               opts=dict(title='Ns_y残差相减', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=loss_3i[-2, 0, 20].detach().cpu().numpy(), win='zres',
        #               opts=dict(title='Ns_z残差相减', colormap='rainbow', xmax=1, xmin=-1))
        #
        # viz_1.heatmap(X=res_x_out[-2, 0, 20].detach().cpu().numpy(), win='xres_out',
        #               opts=dict(title='Ns_x_out_残差', colormap='rainbow', xmax=1, xmin=-1))
        # viz_1.heatmap(X=res_x_tag[-2, 0, 20].detach().cpu().numpy(), win='xres_tag',
        #               opts=dict(title='Ns_x_target_残差', colormap='rainbow', xmax=1, xmin=-1))
        total_outputs=torch.cat(total_outputs,dim=0)
        total_labels=torch.cat(total_labels,dim=0)
        total_l2loss=normal_rmse(total_outputs,total_labels)

        mean_loss=total_loss/len(dataloader)
        mean_loss_nsx=total_loss_nsx/len(dataloader)
        mean_loss_nsy = total_loss_nsy / len(dataloader)
        mean_loss_nsz = total_loss_nsz / len(dataloader)
        mean_loss_nscont = total_loss_nscont / len(dataloader)
        mean_loss_psnr=total_loss_psnr/len(dataloader)
        mean_loss_ssim = total_loss_ssim / len(dataloader)
        # print('eval_success mean_loss=%.6f mean_loss_nsx=%.4f mean_psnr=%.4f ' % (mean_loss, mean_loss_nsx, mean_loss_psnr))
        # print('eval_success mean_loss=%.6f mean_psnr=%.4f mean_loss_ssim%.4f ' % (mean_loss, mean_loss_psnr, mean_loss_ssim))

        LOSSes = {}
        LOSSes['mean_loss']=mean_loss
        LOSSes['mean_loss_nsx'] = mean_loss_nsx
        LOSSes['mean_loss_nsy'] = mean_loss_nsy
        LOSSes['mean_loss_nsz'] = mean_loss_nsz
        LOSSes['mean_loss_nscont'] = mean_loss_nscont
        LOSSes['mean_loss_nspsnr'] = mean_loss_psnr
        LOSSes=str(LOSSes)
    # with open(eval_loss_savefilename,'w') as f:
    #     f.write(LOSSes)

    # viz_1.line([[mean_loss,
    #              mean_loss_nsx,
    #              mean_loss_nsy,
    #              mean_loss_nsz,
    #              mean_loss_nscont,
    #              mean_loss_psnr
    #              ]], [epoch], win='eval_line', update='append',
    #            opts=dict(title='eval_line',
    #                      legend=[ 'loss',
    #                              'NS_loss_X',
    #                              'NS_loss_y',
    #                              'NS_loss_z',
    #                              'NS_loss_cont',
    #                               'loss_psnr'
    #                              ]))
def test(bayes_nn, dataloader,args):
    with torch.no_grad():
        bayes_nn.eval().cuda()
        save_path=args.use_output_savepath
        save_names=os.listdir(args.eval_label_data_dir)
        save_names_group=[]
        print(len(dataloader))
        mini_group_num=int((args.input_t_length+1)/2)
        group_num=len(dataloader)
        name_groups=[]

        for i in range(group_num):
            mini_group=[]
            for j in range(mini_group_num):
                print(i*  args.input_t_length + j)
                mini_group.append(os.path.join(save_path,save_names[i*  args.input_t_length + j]))
                mini_group.append(os.path.join(save_path,save_names[i * args.input_t_length + j][:-4]+'_5.pth'))
            name_groups.append(mini_group[:-1])
        for (input_dict,target_dict),name_mini_group in zip(dataloader,name_groups):
            # move to GPU if available
            target = target_dict['v_oumiga'].to(device)[0][:, :3]
            _,_,d,h,w=(input_dict['v_oumiga'][0]).shape
            min_x=input_dict['x_grid'][0][0,0,0]

            # x_grid=  torch.nn.functional.interpolate(input_dict['x_grid'][0].to(device),size=(2 *d,2 *h,2 *w),
            #                                        mode='trilinear',align_corners=False)
            # y_grid = torch.nn.functional.interpolate(input_dict['y_grid'][0].to(device), size=(2 * d, 2 * h, 2 * w),
            #                                          mode='trilinear', align_corners=False)
            # z_grid = torch.nn.functional.interpolate(input_dict['z_grid'][0].to(device), size=(2 * d, 2 * h, 2 * w),
            #                                          mode='trilinear', align_corners=False)
            x_grid,y_grid,z_grid=grid_upsample(input_dict)
            density = input_dict['density'][0].cuda()
            viscosity =input_dict['viscosity'][0].cuda()
            t_step =input_dict['t_step'][0].cuda()
            input = input_dict['v_oumiga'].to(device)[0].cuda()
            output_is = 0
            steps = (
            target_dict['x_grid'][0].to(device),
            target_dict['y_grid'][0].to(device),
            target_dict['z_grid'][0].to(device))
            # compute model output
            for i in range(len(model)):
                output_i = bayes_nn[i].forward(input, steps, t_step, density, viscosity,t_interp=args.time_interp)
                output_is+=output_i[:, :3]
            outputs=output_is/3
            output_np=outputs.cpu().numpy()
            label_np=target.cpu().numpy()
            loss_rmse,mean = normal_rmse(outputs,target)
            print(loss_rmse/mean)
            for output_i,save_name in zip(outputs,name_mini_group):
                output_dict={}
                output_dict['x_grid']=x_grid.cpu()
                output_dict['y_grid']=y_grid.cpu()
                output_dict['z_grid'] = z_grid.cpu()
                output_dict['t_step'] = t_step
                output_dict['v_oumiga']= output_i
                output_dict['density']=density.cpu()
                output_dict['viscosity']=viscosity.cpu()
                torch.save(output_dict,save_name)
            torch.cuda.empty_cache()
def use(bayes_nn, dataloader,args):
    with torch.no_grad():
        bayes_nn.eval().cuda()
        save_path=args.use_output_savepath
        save_names=os.listdir(args.eval_label_data_dir)
        save_names_group=[]
        print(len(dataloader))
        mini_group_num=int(args.input_t_length)
        group_num=len(dataloader)
        name_groups=[]
        if args.time_interp:
            for i in range(group_num):
                mini_group=[]
                for j in range(mini_group_num):
                    mini_group.append(os.path.join(save_path,save_names[i*  args.input_t_length + j]))
                    mini_group.append(os.path.join(save_path,save_names[i * args.input_t_length + j][:-4]+'_5.pth'))
                name_groups.append(mini_group[:-1])
        else:
            for i in range(group_num):
                mini_group=[]
                for j in range(mini_group_num):
                    mini_group.append(os.path.join(save_path,save_names[i*  args.input_t_length + j]))
                name_groups.append(mini_group)
        for (input_dict,target_dict),name_mini_group in zip(dataloader,name_groups):
            # move to GPU if available
            target = target_dict['v_oumiga'].to(device)[0][:, :3]
            _,_,d,h,w=(target_dict['v_oumiga'][0]).shape
            min_x=target_dict['x_grid'][0][0,0,0]

            # x_grid=  torch.nn.functional.interpolate(target_dict['x_grid'][0].to(device),size=(2 *d,2 *h,2 *w),
            #                                        mode='trilinear',align_corners=False)
            # y_grid = torch.nn.functional.interpolate(target_dict['y_grid'][0].to(device), size=(2 * d, 2 * h, 2 * w),
            #                                          mode='trilinear', align_corners=False)
            # z_grid = torch.nn.functional.interpolate(target_dict['z_grid'][0].to(device), size=(2 * d, 2 * h, 2 * w),
            #                                          mode='trilinear', align_corners=False)
            x_grid,y_grid,z_grid=grid_upsample(target_dict)
            steps=(x_grid.unsqueeze(dim=0).unsqueeze(dim=0).cuda(),
                   y_grid.unsqueeze(dim=0).unsqueeze(dim=0).cuda(),
                   z_grid.unsqueeze(dim=0).unsqueeze(dim=0).cuda())
            density = target_dict['density'][0].cuda()
            viscosity =target_dict['viscosity'][0].cuda()
            t_step =target_dict['t_step'][0].cuda()
            input = target_dict['v_oumiga'].to(device)[0].cuda()
            output_is = 0
            # compute model output
            for i in range(len(model)):
                output_i = bayes_nn[i].forward(input, steps, t_step, density, viscosity,t_interp=args.time_interp)
                output_is+=output_i[:, :3]
            outputs=output_is/3


            for output_i,save_name in zip(outputs,name_mini_group):
                output_dict={}
                output_dict['x_grid']=x_grid.cpu()
                output_dict['y_grid']=y_grid.cpu()
                output_dict['z_grid'] = z_grid.cpu()
                output_dict['t_step'] = t_step
                output_dict['v_oumiga']= output_i
                output_dict['density']=density.cpu()
                output_dict['viscosity']=viscosity.cpu()
                torch.save(output_dict,save_name)
            torch.cuda.empty_cache()
# def cubic_use_matlab(x_line,y_line,z_line,U,V,W,x_new_line,y_new_line,z_new_line):
#
#
#     U_matlab=  matlab.double(U.tolist())
#     V_matlab = matlab.double(V.tolist())
#     W_matlab = matlab.double(W.tolist())
#
#
#     U_up= np.asarray(  eng.interp3(x_line,y_line,z_line,  U_matlab,    y_new_line, z_new_line, x_new_line,'cubic'))
#     U_up = U_up.reshape(64, 64, 64)
#     plt.subplot(121)
#     plt.imshow(U[0], cmap='jet')
#     plt.colorbar()
#     plt.subplot(122)
#     plt.imshow(U_up[0], cmap='jet')
#     plt.colorbar()
#     plt.show()
#
#
#     V_up =np.asarray(  eng.interp3(x_line, y_line, z_line, V_matlab, y_new_line, z_new_line, x_new_line, 'cubic'))
#     W_up =np.asarray(  eng.interp3(x_line, y_line, z_line, W_matlab, y_new_line, z_new_line, x_new_line, 'cubic'))
#
#
#     return U_up,V_up,W_up
# def evaluate_cubic(bayes_nn, dataloader,eval_loss_savefilename):
#     # set model to evaluation mode
#     bayes_nn.eval().cuda()
#
#     # summary for current eval loop
#     summ = []
#     count = 0
#     # compute metrics over the dataset
#     total_loss = 0.
#     total_loss_psnr=0.
#     total_loss_nsx=0.
#     total_loss_nsy = 0.
#     total_loss_nsz = 0.
#     total_loss_nscont = 0.
#
#     for i, (input_dict, target_dict) in enumerate(dataloader):
#         # move to GPU if available
#         loss_nsx=0.
#         loss_nsy = 0.
#         loss_nsz = 0.
#         loss_nscont = 0.
#         loss_m_show = 0.
#         d, h, w = (input_dict['x_grid'][0,0,0]).shape
#         l=max(d,h,w)
#         # d,h,w=(steps[0]).shape
#         # x_line = steps[0][0, 0, :]
#         # y_line = steps[1][0, :, 0].reshape(1,-1)[0]
#         # z_line = steps[2][:, 0, 0].reshape(1,-1)[0]
#         # x_sta = x_line[0]
#         # x_end = x_line[-1]
#         # x_step= x_line[1]-x_line[0]
#         # y_sta = y_line[0]
#         # y_end = y_line[-1]
#         # y_step =y_line[1] - y_line[0]
#         # z_sta = z_line[0]
#         # z_end = z_line[-1]
#         # z_step = z_line[1] - z_line[0]
#         # x_new_line = np.linspace(x_sta-0.25*x_step, x_end+0.25*x_step, len(x_line) * args.upsample_index)
#         # y_new_line = np.linspace(y_sta-0.25*y_step, y_end+0.25*y_step, len(y_line) * args.upsample_index)
#         # z_new_line = np.linspace(z_sta-0.25*z_step, z_end+0.25*z_step, len(z_line) * args.upsample_index)
#         # Y_new, Z_new, X_new = np.meshgrid(y_new_line, z_new_line, x_new_line)
#         # X_new_line = X_new.reshape(1, -1)[0]
#         # Y_new_line = Y_new.reshape(1, -1)[0]
#         # Z_new_line = Z_new.reshape(1, -1)[0]
#         x = np.linspace(0, (l-1)/100, l)
#         y = np.linspace(0, (l-1)/100, l)
#         z = np.linspace(0, (l-1)/100, l)
#
#         Y, Z, X = np.meshgrid(y, z, x)
#         x1 = np.linspace(0,(l-1)/100, opt.upsample_index*l)
#         y1 = np.linspace(0,(l-1)/100, opt.upsample_index*l)
#         z1 = np.linspace(0,(l-1)/100, opt.upsample_index*l)
#         ynew_, znew, xnew = np.meshgrid(y1, z1, x1)
#         xnew = xnew.reshape(1, -1)[0]
#         ynew = ynew_.reshape(1, -1)[0]
#         znew = znew.reshape(1, -1)[0]
#
#         # 定义输入数据
#         x = matlab.double(x.tolist())
#         y = matlab.double(y.tolist())
#         z = matlab.double(z.tolist())
#
#
#         xnew = matlab.double(xnew.tolist())
#         ynew = matlab.double(ynew.tolist())
#         znew = matlab.double(znew.tolist())
#
#         input =torch.nn.functional.interpolate(input_dict['v_oumiga'][0],size=(l,l,l),mode='trilinear')
#
#         U_up_t=[]
#         V_up_t = []
#         W_up_t = []
#
#         for i in range(input.shape[0]):
#             U = input[i,0]
#             V = input[i,1]
#             W = input[i,2]
#             if i >= 1:
#                 U_up_last = U_up
#                 V_up_last = V_up
#                 W_up_last = W_up
#             U_matlab = matlab.double(U.tolist())
#             U_up = np.asarray(
#                 eng.interp3(x, y, z,U_matlab, ynew,znew,xnew,opt.interp_form))
#             U_up = U_up.reshape( opt.upsample_index*l,  opt.upsample_index*l,  opt.upsample_index*l)
#
#             V_matlab = matlab.double(V.tolist())
#             V_up = np.asarray(
#                 eng.interp3(x, y, z,V_matlab, ynew,znew,xnew,opt.interp_form))
#             V_up = V_up.reshape( opt.upsample_index*l,  opt.upsample_index*l,  opt.upsample_index*l)
#
#             W_matlab = matlab.double(W.tolist())
#             W_up = np.asarray(
#                 eng.interp3(x, y, z,W_matlab, ynew,znew,xnew,opt.interp_form))
#             W_up = W_up.reshape( opt.upsample_index*l,  opt.upsample_index*l,  opt.upsample_index*l)
#
#             # plt.subplot(121)
#             # plt.imshow(U[:, :, 10], cmap='jet')
#             # plt.colorbar()
#             # plt.subplot(122)
#             # plt.imshow(U_up[:, :, 20], cmap='jet')
#             # plt.colorbar()
#             # plt.show()
#             #
#             # plt.subplot(121)
#             # plt.imshow(V[:, :, 10], cmap='jet')
#             # plt.colorbar()
#             # plt.subplot(122)
#             # plt.imshow(V_up[:, :, 20], cmap='jet')
#             # plt.colorbar()
#             # plt.show()
#             #
#             # plt.subplot(121)
#             # plt.imshow(W[:, :, 10], cmap='jet')
#             # plt.colorbar()
#             # plt.subplot(122)
#             # plt.imshow(W_up[:, :, 20], cmap='jet')
#             # plt.colorbar()
#             # plt.show()
#
#
#             U_up_t.append(U_up)
#             V_up_t.append(V_up)
#             W_up_t.append(W_up)
#             if i>=1:
#                 U_up_t.append((U_up + U_up_last) / 2)
#                 V_up_t.append((V_up + V_up_last) / 2)
#                 W_up_t.append((W_up + W_up_last) / 2)
#         U_up_t=  np.stack(U_up_t,axis=0)
#         V_up_t = np.stack(V_up_t, axis=0)
#         W_up_t = np.stack(W_up_t, axis=0)
#
#         output=np.stack([U_up_t,V_up_t,W_up_t],axis=1)
#         output=torch.from_numpy(output).cuda()
#         output=torch.nn.functional.interpolate(output,size=(opt.upsample_index*d,opt.upsample_index*h,opt.upsample_index*w),
#                                                mode='trilinear')
#
#         steps = (
#         target_dict['x_grid'][0].to(device), target_dict['y_grid'][0].to(device), target_dict['z_grid'][0].to(device))
#         density = target_dict['density'][0]
#         viscosity = target_dict['viscosity'][0]
#         t_step = target_dict['t_step'][0]
#         input = input_dict['v_oumiga'].to(device)[0]
#         target = target_dict['v_oumiga'].to(device)[0]
#
#
#         log_eq_i, loss_1i, loss_2i, loss_3i, loss_4i, res_x_out, res_x_tag = bayes_nn.criterion_grid(output,
#                                                                                                           target,
#                                                                                                           steps,
#                                                                                                           t_step,
#                                                                                                           density,
#                                                                                                           viscosity,
#                                                                                                           i, 1)
#         loss_nsx, loss_nsy, loss_nsz, loss_nscont = monitor(loss_nsx, loss_nsy, loss_nsz,
#                                                        loss_nscont, loss_1i.detach(), loss_2i.detach(),
#                                                        loss_3i.detach(), loss_4i.detach())
#         loss_rmse,mean = normal_rmse(output,target)
#         loss_rmse=loss_rmse/mean
#         loss_psnr=psnr(output[:, :3].cpu().numpy(), target[:, :3].cpu().numpy())
#
#         loss_nsx=  loss_nsx /len(model)
#         loss_nsy = loss_nsy / len(model)
#         loss_nsz = loss_nsz / len(model)
#         loss_nscont=loss_nscont / len(model)
#         total_loss_psnr+=loss_psnr
#         total_loss+=loss_rmse
#         total_loss_nsx+=loss_nsx.cpu().numpy()
#         total_loss_nsy+=loss_nsy.cpu().numpy()
#         total_loss_nsz+=loss_nsz.cpu().numpy()
#         total_loss_nscont+=loss_nscont.cpu().numpy()
#     _,_,d,h,w=target.shape
#     mean_loss=total_loss/len(dataloader)
#     mean_loss_nsx=total_loss_nsx/len(dataloader)
#     mean_loss_nsy = total_loss_nsy / len(dataloader)
#     mean_loss_nsz = total_loss_nsz / len(dataloader)
#     mean_loss_nscont = total_loss_nscont / len(dataloader)
#     mean_loss_psnr=total_loss_psnr/len(dataloader)
#     print('eval_success mean_loss=%.6f mean_loss_nsx=%.4f mean_psnr=%.4f ' % (mean_loss, mean_loss_nsx, mean_loss_psnr))
#     LOSSes = {}
#     LOSSes['mean_loss']=mean_loss
#     LOSSes['mean_loss_nsx'] = mean_loss_nsx
#     LOSSes['mean_loss_nsy'] = mean_loss_nsy
#     LOSSes['mean_loss_nsz'] = mean_loss_nsz
#     LOSSes['mean_loss_nscont'] = mean_loss_nscont
#     LOSSes['mean_loss_nspsnr'] = mean_loss_psnr
#     LOSSes=str(LOSSes)
#     with open(eval_loss_savefilename,'w') as f:
#         f.write(LOSSes)
#
#     # viz_1.line([[mean_loss,
#     #              mean_loss_nsx,
#     #              mean_loss_nsy,
#     #              mean_loss_nsz,
#     #              mean_loss_nscont,
#     #              mean_loss_psnr
#     #              ]], [epoch], win='eval_line', update='append',
#     #            opts=dict(title='eval_line',
#     #                      legend=[ 'loss',
#     #                              'NS_loss_X',
#     #                              'NS_loss_y',
#     #                              'NS_loss_z',
#     #                              'NS_loss_cont',
#     #                               'loss_psnr'
#     #                              ]))
# def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
#                        restore_file=None):
#     # reload weights from restore_file if specified
#     if restore_file is not None:
#         restore_path = os.path.join(save_path, args.trained_para)
#         logging.info("Restoring parameters from {}".format(restore_path))
#         utils.load_checkpoint(restore_path, model, optimizer)
#
#     best_val_acc = 0.0
#
#     '''
#     # train add logger,epoch two parameters
#     '''
#
#     # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
#     # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#
#     for epoch in range(opt.start_epoch,opt.epochs):
#         # Run one epoch
#         #         scheduler.step()
#         logging.info("Epoch {}/{}".format(epoch + 1, opt.epochs))
#
#         # compute number of batches in one epoch (one full pass over the training set)
#         train(model, optimizer, loss_fn, train_dataloader, metrics, epoch)
#         # Evaluate for one epoch on validation set
#         val_metrics = evaluate(model, loss_fn, val_dataloader, metrics)
#
#         val_acc = val_metrics['PSNR']
#         is_best = val_acc >= best_val_acc
#
#         # Save weights
#         # utils.save_checkpoint({'epoch': epoch + 1,
#         #                        'state_dict': model.state_dict(),
#         #                        'optim_dict': optimizer.state_dict()},
#         #                       is_best=is_best,
#         #                       checkpoint=model_dir)
#
#         # If best_eval, best_save_path
#         if is_best:
#             logging.info("- Found new best accuracy")
#             best_val_acc = val_acc
#             savefilename = save_path + '/finetune_' + str(epoch) + '.pth'
#             torch.save(model.state_dict(),savefilename)
#
#             # Save best val metrics in a json file in the model directory
#         #     best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
#         #     utils.save_dict_to_json(val_metrics, best_json_path)
#         #
#         # # Save latest val metrics in a json file in the model directory
#         # last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
#         # utils.save_dict_to_json(val_metrics, last_json_path)
#
#     plt.plot(global_loss)
#     plt.savefig("final loss.jpg")

def monitor( loss_1, loss_2, loss_3, loss_4, loss_1i, loss_2i, loss_3i, loss_4i):
    #loss_f = nn.MSELoss()
    loss_1 += loss_fn(loss_1i,torch.zeros_like(loss_1i))
    loss_2 += loss_fn(loss_2i,torch.zeros_like(loss_2i))
    loss_3 += loss_fn(loss_3i,torch.zeros_like(loss_3i))
    loss_4 += loss_fn(loss_4i, torch.zeros_like(loss_4i))
    return loss_1, loss_2, loss_3, loss_4
def loss_fn(outputs, labels):
    result_by = torch.sqrt((outputs - labels).pow(2).mean())
    return result_by
def normal_rmse(output,label):
    loss_rmse=torch.sqrt(((output[:,0]-label[:,0]).pow(2)+(output[:,1]-label[:,1]).pow(2)+(output[:,2]-label[:,2]).pow(2)).mean()).cpu().numpy()
    mean=torch.sqrt((label[:,0].pow(2)+label[:,1].pow(2)+label[:,2].pow(2)).mean()).cpu().numpy()
    return loss_rmse,mean


def accuracy(outputs, labels):
    nume = np.max(outputs, axis=(1, 2, 3), keepdims=True)  # (N,)
    deno = np.sum((outputs - labels) ** 2, axis=(1, 2, 3),
                  keepdims=True)  # (N,)

    psnr = 10 * np.sum(np.log((nume * 256) ** 2 / deno) / math.log(10)) / outputs.shape[0]
    return psnr


def calculate_psnr(outputs, labels):
    psnr = np.log((np.max(outputs) / np.sum((outputs - labels) ** 2) /
                   outputs.shape[0])) / math.log(10)
    return psnr

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    max=np.max(img2)
    mean_=np.mean(img2)
    if mse == 0:
        return float('inf')
    else:
        ps=20*np.log10(max/np.sqrt(mse))
        return ps


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    torch.cuda.manual_seed(230)
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'D:\desktop\\3DFlowfield_reconstruction\\NS_RFDN_GRU_3D\logs\\logs.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
#    train_dataloaders = fetch_dataloader(['train'], args.label_data_dir,args.input_t_length,opt.batch_size,downsample=args.downsample_form)
    eval_dataloaders = fetch_dataloader(['val'], args.eval_label_data_dir, args.input_t_length, opt.batch_size,
                                         downsample=args.downsample_form,
                                        downsample_index=args.upsample_index,time_interp=args.time_interp)

  #  train_dl = train_dataloaders['train']
    val_dl = eval_dataloaders['val']
    use_dl=fetch_dataloader(['use'], args.eval_label_data_dir, args.input_t_length, opt.batch_size,
                                         downsample=None,downsample_index=args.upsample_index,time_interp=args.time_interp)
    use_dl=use_dl['use']


    logging.info("- done.")


    model =torch.nn.ModuleList([RFDN(upscale=2, start_iter_num=args.start_iter_num,NS_ECA=args.NS_ECA).to(device),
                                RFDN(upscale=2, start_iter_num=args.start_iter_num,NS_ECA=args.NS_ECA).to(device),
                                RFDN(upscale=2, start_iter_num=args.start_iter_num,NS_ECA=args.NS_ECA).to(device),
])
    if args.use_bayes:
        bayes_nn=BayesNN(model, n_samples=len(model)  , noise=opt.var_eq).to(device)
        if args.trained_para is not None:
            data = torch.load(args.trained_para)
            model_dict = bayes_nn.state_dict()
            state_dict = {}
            for k, v in data.items():
                if k in model_dict.keys():
                    state_dict[k] = v
            # state_dict={k:v for k,v in data.items() if k in model_dict.keys()}
            # state_dict['en.Subp.netMain.0.weight']=torch.randn([128,130,3,3],requires_grad=True,dtype=torch.float32)
            model_dict.update(state_dict)
            bayes_nn.load_state_dict(model_dict)
            loss_savefilename = opt.loss_save_path + '/LOSS_%04d.pth' % (500)
            eval_loss_savefilename=opt.eval_label_data_dir + '/eval_LOSS_%04d.txt' % (500)
            if args.test_or_eval_or_use=='eval':
                evaluate(bayes_nn, val_dl, eval_loss_savefilename)
            elif args.test_or_eval_or_use=='use':
                use(bayes_nn,val_dl,args)
            elif args.test_or_eval_or_use=='test':
                test(bayes_nn,val_dl,args)

       # evaluate_cubic(bayes_nn,val_dl,eval_loss_savefilename=opt.eval_label_data_dir + '/cubic_LOSS_%04d.txt' % (500))

