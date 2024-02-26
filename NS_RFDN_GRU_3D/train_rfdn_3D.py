"""Train the model"""
import argparse
import logging
import os
import math
import numpy as np
import time
import visdom
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
from N_S_loss_Fan_init_coord import cal_NS_residual_vv_rotate
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from model_RFDN.RFDN_3D_downsample_GRU_input_divide import RFDN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ssim import ssim
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./parameters', help="Directory containing params.json")
parser.add_argument('--label_data_dir', default=' ', help="label_data_dir")
parser.add_argument('--eval_label_data_dir', default=' ', help="")
parser.add_argument('--para_save_path', default='',
                    help="")
parser.add_argument('--trained_para',default='')
parser.add_argument('--start_epoch',default=1,help='')#'预训练权重地址')
parser.add_argument('--loss_save_path', default='',
                    help="")
parser.add_argument('--NS_ECA',default=1,help='Use_NS_ECA')
parser.add_argument('--use_bayes',default=1,help='use_bayes')
parser.add_argument('--input_t_length', default=5, help="the length of per input group")
parser.add_argument('--var_eq',default=1e-4,help='bayes ')
parser.add_argument('--downsample_form',default='mean',help='mean/gauss')
parser.add_argument('--upsample_index',default=2,help='upsample_index')
parser.add_argument('--time_interp',default=0,help='time_interp')

parser.add_argument('--optim', default='adam')
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--batch_size', type=int, default=1 ,help='order to 1')
parser.add_argument('--start_iter_num', default=1,help='order to 1')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lrf', type=float, default=0.001)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()
#net = get_paras().to(device)
# python -m visdom.server


global_loss = []
save_path = opt.para_save_path
def train(model, optimizers, schedulers, dataloader, epoch,model_savename,loss_savename):
    # set model to training mode
    model.train()
    # print('this the type of the model', type(model))
    # summary for current training loop and a running average object for loss
    rmse_show = 0.
    loss_1_show = 0.
    loss_2_show = 0.
    loss_3_show = 0.
    loss_4_show = 0.
    loss_m_show = 0.
    LOSS = []
    LOSS1 = []
    LOSS2 = []
    LOSS3 = []
    LOSS4 = []
    # Use tqdm for progress bar
    viz_1 = visdom.Visdom()
    for batch_idx, (input_dict, target_dict) in enumerate(dataloader):
        # move to GPU if available
        loss_1, loss_2, loss_3, loss_4 = 0, 0, 0, 0
        steps = (
        target_dict['x_grid'][0].to(device).float(),
        target_dict['y_grid'][0].to(device).float(),
        target_dict['z_grid'][0].to(device).float())
        density = target_dict['density'][0]
        viscosity = target_dict['viscosity'][0].float()
        t_step = target_dict['t_step'][0].float()
        input = input_dict['v_oumiga'].to(device)[0].float()
        target = target_dict['v_oumiga'].to(device)[0].float()
        output = torch.zeros_like(target).to(device)
        output_is = []
        t, C, Hd, Hw, Hh = target.shape

        for i in range(3):
            optimizers[i].zero_grad()
            output_i=model[i].forward(input,steps, t_step, density, viscosity)
            output_is.append(output_i)
            loss_data = torch.nn.MSELoss()
            loss_data=loss_data(output_i, target)
            loss_NS,loss_1i,loss_2i,loss_3i,loss_4i,res_x_out,res_x_tag=criterion_grid(output_i, target, steps, t_step, density, viscosity)
            input_up = torch.nn.functional.interpolate(input, (Hd, Hw, Hh), mode='trilinear', align_corners=True)
            input_up_1 = (input_up[0] + input_up[1]) / 2
            input_up_3 = (input_up[1] + input_up[2]) / 2
            input_up = torch.stack([input_up[0], input_up_1, input_up[1], input_up_3, input_up[2]])
            _, res_x_input_sub, _, _, _, res_x_input, _ = criterion_grid(input_up, target, steps, t_step,
                                                                                       density, viscosity)
            loss_1, loss_2, loss_3, loss_4 = monitor(loss_1, loss_2, loss_3,
                                                           loss_4, loss_1i.detach(), loss_2i.detach(),
                                                           loss_3i.detach(), loss_4i.detach())
            back_loss=loss_data+loss_NS*0.001
            # clear previous gradients, compute gradients of all variables wrt loss
            back_loss.backward()
            optimizers[i].step()
            schedulers[i].step(epoch)
            output += output_i.detach()
        output=output/3
        loss_rmse,mean=normal_rmse(output, target)
        rmse = loss_rmse / mean
        rmse_show += rmse
        input_show = nn.functional.interpolate(input, size=(Hd, Hw, Hh), mode='trilinear', align_corners=False)
        if batch_idx % 10 == 0:
            viz_1.heatmap(X=target[-2, 0, 20].cpu().numpy(), win='ulabel',
                          opts=dict(title='ulabel', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=output[-2, 0, 20].cpu().numpy(), win='uout',
                          opts=dict(title='uout1', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=target[-2, 1, 20].cpu().numpy(), win='vlabel',
                          opts=dict(title='vlabel', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=output[-2, 1, 20].cpu().numpy(), win='vout',
                          opts=dict(title='vout1', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=target[-2, 2, 20].cpu().numpy(), win='wlabel',
                          opts=dict(title='wlabel', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=output[-2, 2, 20].cpu().numpy(), win='wout',
                          opts=dict(title='wout1', colormap='rainbow', xmax=1, xmin=-1))

            viz_1.heatmap(X=input_show[-1, 0, 20].cpu().numpy(), win='uinput',
                          opts=dict(title='uinput', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=input_show[-1, 1, 20].cpu().numpy(), win='vinput',
                          opts=dict(title='vinput', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=input_show[-1, 2, 20].cpu().numpy(), win='winput',
                          opts=dict(title='winput', colormap='rainbow', xmax=1, xmin=-1))

            viz_1.heatmap(X=input_show[-1, 0, 20].cpu().numpy() - target[-1, 0, 20].cpu().numpy(), win='uinput-res',
                          opts=dict(title='uinput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
            viz_1.heatmap(X=input_show[-1, 1, 20].cpu().numpy() - target[-1, 1, 20].cpu().numpy(), win='vinput-res',
                          opts=dict(title='vinput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
            viz_1.heatmap(X=input_show[-1, 2, 20].cpu().numpy() - target[-1, 2, 20].cpu().numpy(), win='winput-res',
                          opts=dict(title='winput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))

            viz_1.heatmap(X=target[-2, 0, 20].cpu().numpy() - output[-2, 0, 20].cpu().numpy(), win='ures',
                          opts=dict(title='u残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
            viz_1.heatmap(X=target[-2, 1, 20].cpu().numpy() - output[-2, 1, 20].cpu().numpy(), win='vres',
                          opts=dict(title='v残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
            viz_1.heatmap(X=target[-2, 2, 20].cpu().numpy() - output[-2, 2, 20].cpu().numpy(), win='wres',
                          opts=dict(title='w残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
            viz_1.heatmap(X=loss_1i[-2, 0, 20].detach().cpu().numpy(), win='xres',
                          opts=dict(title='Ns_x残差相减', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=loss_2i[-2, 0, 20].detach().cpu().numpy(), win='yres',
                          opts=dict(title='Ns_y残差相减', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=loss_3i[-2, 0, 20].detach().cpu().numpy(), win='zres',
                          opts=dict(title='Ns_z残差相减', colormap='rainbow', xmax=1, xmin=-1))

            viz_1.heatmap(X=res_x_out[-2, 0, 20].detach().cpu().numpy(), win='xres_out',
                          opts=dict(title='Ns_x_out_残差', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=res_x_tag[-2, 0, 20].detach().cpu().numpy(), win='xres_tag',
                          opts=dict(title='Ns_x_target_残差', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=res_x_input[-2, 0, 20].detach().cpu().numpy(), win='xres_in',
                          opts=dict(title='Ns_x_input_残差', colormap='rainbow', xmax=1, xmin=-1))
            viz_1.heatmap(X=res_x_input_sub[-2, 0, 20].detach().cpu().numpy(), win='xres_in_sub',
                          opts=dict(title='Ns_x_input_相减', colormap='rainbow', xmax=1, xmin=-1))
        loss_1 /= 3
        loss_2 /= 3
        loss_3 /= 3
        loss_4 /= 3
        loss = (loss_1.cpu().numpy() + loss_2.cpu().numpy() + loss_3.cpu().numpy() + loss_4.cpu().numpy()) / 4
        loss_1_show += loss_1.cpu().numpy()
        loss_2_show += loss_2.cpu().numpy()
        loss_3_show += loss_3.cpu().numpy()
        loss_4_show += loss_4.cpu().numpy()
        loss_m_show += loss
        LOSS.append(loss)
        LOSS1.append(loss_1.cpu().numpy())
        LOSS2.append(loss_2.cpu().numpy())
        LOSS3.append(loss_3.cpu().numpy())
        LOSS4.append(loss_4.cpu().numpy())
        print(
        'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg_N-S_Loss: {:.6f} Normal RMSE_loss: {:.5f}, learning_rate*1e5={:.2f}'.format(
            epoch, batch_idx, len(dataloader),
            100. * batch_idx / len(dataloader), loss, rmse,
            1e5 * schedulers[0].optimizer.param_groups[0]['lr']))
        if batch_idx % 20 == 0:
            iter_num = batch_idx + (epoch) * len(dataloader)
            viz_1.line([[loss_m_show / (batch_idx + 1),
                         loss_1_show / (batch_idx + 1),
                         loss_2_show / (batch_idx + 1),
                         loss_3_show / (batch_idx + 1),
                         loss_4_show / (batch_idx + 1),
                         rmse ]], [iter_num], win='training_line', update='append',
                       opts=dict(title='training_line',
                                 legend=['mean_NS_loss',
                                         'NS_loss_X',
                                         'NS_loss_y',
                                         'NS_loss_z',
                                         'NS_loss_cont',
                                         'rmse'
                                         ]))
    rmse_train = rmse_show / len(dataloader)
    mean_LOSS1 = np.array(LOSS1).mean()
    mean_LOSS2 = np.array(LOSS2).mean()
    mean_LOSS3 = np.array(LOSS3).mean()
    mean_LOSS4 = np.array(LOSS4).mean()
    LOSSes = {}
    LOSSes['rmse'] = rmse_train
    LOSSes['LOSS1'] = mean_LOSS1
    LOSSes['LOSS2'] = mean_LOSS2
    LOSSes['LOSS3'] = mean_LOSS3
    LOSSes['LOSS4'] = mean_LOSS4
    # torch.save(LOSSes, loss_savename)
    viz_1.line([[mean_LOSS1,
                 mean_LOSS2,
                 mean_LOSS3,
                 mean_LOSS4,
                 rmse_train,
                 ]],
               [epoch], win='training_line_epoch', update='append', opts=dict(title='training_line',
                                                                              legend=[
                                                                                  'NS_loss_X',
                                                                                  'NS_loss_y',
                                                                                  'NS_loss_z',
                                                                                  'NS_loss_cont',
                                                                                  'rmse'
                                                                              ]))
    para_savefilename = model_savename + '/finetune_%04d.pth' % (epoch)
    if epoch % 1 == 0:
        model.eval()
        torch.save(model.state_dict(),para_savefilename)


def loss_fun(output,target):
    result_by = (torch.mean(abs(output - target)))
    result_by =torch.sqrt((output - target).pow(2).mean())
    # out=output.detach().cpu().numpy()
    # out_1=np.mean(np.mean(out,axis=4),axis=3)
    #result_by = (torch.mean((output - target) ** 2) ).sqrt()
    return result_by
def monitor( loss_1, loss_2, loss_3, loss_4, loss_1i, loss_2i, loss_3i, loss_4i):
    loss_1 += loss_fun(loss_1i,torch.zeros_like(loss_1i))
    loss_2 += loss_fun(loss_2i,torch.zeros_like(loss_2i))
    loss_3 += loss_fun(loss_3i,torch.zeros_like(loss_3i))
    loss_4 += loss_fun(loss_4i, torch.zeros_like(loss_4i))
    return loss_1, loss_2, loss_3, loss_4


def criterion_grid(output,target,steps, t_step, density, viscosity):
    u = output[:, 0].unsqueeze(dim=1)
    v = output[:, 1].unsqueeze(dim=1)
    w = output[:, 2].unsqueeze(dim=1)
    u_target = target[:, 0].unsqueeze(dim=1)
    v_target = target[:, 1].unsqueeze(dim=1)
    w_target = target[:, 2].unsqueeze(dim=1)
    x_step, y_step, z_step = steps
    res_x_out, res_y_out, res_z_out, res_cont_out = cal_NS_residual_vv_rotate(u, v, w,
                                                                              x_step, y_step, z_step, t_step, density,
                                                                              viscosity)
    res_x_tag, res_y_tag, res_z_tag, res_cont_tag = cal_NS_residual_vv_rotate(u_target, v_target, w_target,
                                                                              x_step, y_step, z_step, t_step,
                                                                              density, viscosity)
    res_x = res_x_out - res_x_tag
    res_y = (res_y_out - res_y_tag)
    res_z = (res_z_out - res_z_tag)
    res_cont = (res_cont_out - res_cont_tag)

    ns_loss=res_x.pow(2).mean()+res_y.pow(2).mean()+res_z.pow(2).mean()+res_cont.pow(2).mean()
    return ns_loss,res_x,res_y,res_z,res_cont,res_x_out,res_x_tag

def evaluate(bayes_nn, dataloader,eval_loss_savefilename,epoch,viz_1):
    # set model to evaluation mode
    with torch.no_grad():
        bayes_nn.eval().cuda()
        # summary for current eval loop
        summ = []
        count = 0
        # compute metrics over the dataset
        total_loss = 0.
        total_loss_psnr=0.
        total_loss_nsx=0.
        total_loss_nsy = 0.
        total_loss_nsz = 0.
        total_loss_nscont = 0.
        loss_fft_show=0.
        for i, (input_dict, target_dict) in enumerate(dataloader):
            # move to GPU if available
            loss_nsx=0.
            loss_nsy = 0.
            loss_nsz = 0.
            loss_nscont = 0.
            loss_m_show = 0.
            steps = (
            target_dict['x_grid'][0].to(device).float(),
            target_dict['y_grid'][0].to(device).float(),
            target_dict['z_grid'][0].to(device).float())
            density = target_dict['density'][0]
            viscosity = target_dict['viscosity'][0].to(device).float()
            t_step = target_dict['t_step'][0].to(device).float()
            input = input_dict['v_oumiga'].to(device)[0].float()
            target = target_dict['v_oumiga'].to(device)[0].float()
            output_is = []
            # compute model output
            output = torch.zeros_like(target)
            for i in range(3):
                output_i = bayes_nn[i].forward(input, steps, t_step, density, viscosity,t_interp=args.time_interp)
                output_is.append(output_i)
                log_eq_i, loss_1i, loss_2i, loss_3i, loss_4i, res_x_out, res_x_tag = bayes_nn.criterion_grid(output_i,
                                                                                                                  target,
                                                                                                                  steps,
                                                                                                                  t_step,
                                                                                                                  density,
                                                                                                                  viscosity,
                                                                                                                  i, 1)

                output += output_i.detach()
                loss_nsx, loss_nsy, loss_nsz, loss_nscont = monitor(loss_nsx, loss_nsy, loss_nsz,
                                                               loss_nscont, loss_1i.detach(), loss_2i.detach(),
                                                               loss_3i.detach(), loss_4i.detach())
            output = output / 3
            loss_rmse,mean = normal_rmse(output,target)

            loss_rmse=loss_rmse/mean
            print(loss_rmse)
            output_fft = torch.log10(abs(torch.fft.fftn(output[:, :3].cpu(), norm='forward')))  # .cpu().detach().numpy()
            target_fft = torch.log10(abs(torch.fft.fftn(target[:, :3].cpu(), norm='forward') ))  # .cpu().detach().numpy()
            fft_rmse,fft_mean=normal_rmse(output_fft,target_fft)
            fft_rmse=fft_rmse/fft_mean
            loss_fft_show+=fft_rmse

            output_fft=output_fft.detach().cpu().numpy()
            target_fft=target_fft.detach().cpu().numpy()
            loss_psnr=psnr(output[:, :3].cpu().numpy(), target[:, :3].cpu().numpy())
            loss_nsx=  loss_nsx /3
            loss_nsy = loss_nsy / 3
            loss_nsz = loss_nsz / 3
            loss_nscont=loss_nscont / 3
            total_loss_psnr+=loss_psnr
            total_loss+=loss_rmse
            total_loss_nsx+=loss_nsx.cpu().numpy()
            total_loss_nsy+=loss_nsy.cpu().numpy()
            total_loss_nsz+=loss_nsz.cpu().numpy()
            total_loss_nscont+=loss_nscont.cpu().numpy()
        mean_loss=total_loss/len(dataloader)
        mean_loss_nsx=total_loss_nsx/len(dataloader)
        mean_loss_nsy = total_loss_nsy / len(dataloader)
        mean_loss_nsz = total_loss_nsz / len(dataloader)
        mean_loss_nscont = total_loss_nscont / len(dataloader)
        mean_loss_psnr=total_loss_psnr/len(dataloader)
        mean_loss_fft=loss_fft_show/len(dataloader)
        LOSSes = {}
        LOSSes['mean_loss']=mean_loss
        LOSSes['mean_loss_nsx'] = mean_loss_nsx
        LOSSes['mean_loss_nsy'] = mean_loss_nsy
        LOSSes['mean_loss_nsz'] = mean_loss_nsz
        LOSSes['mean_loss_nscont'] = mean_loss_nscont
        LOSSes['mean_loss_nspsnr'] = mean_loss_psnr
        LOSSes=str(LOSSes)
        with open(eval_loss_savefilename,'w') as f:
            f.write(LOSSes)
        print('eval_success mean_loss=%.4f mean_loss_nsx=%.4f'%(mean_loss,mean_loss_nsx))
        viz_1.line([[mean_loss,
                     mean_loss_nsx,
                     mean_loss_nsy,
                     mean_loss_nsz,
                     mean_loss_nscont,
                     mean_loss_psnr,
                     mean_loss_fft
                     ]], [epoch], win='eval_line_', update='append',
                   opts=dict(title='eval_line',
                             legend=[ 'loss',
                                     'NS_loss_X',
                                     'NS_loss_y',
                                     'NS_loss_z',
                                     'NS_loss_cont',
                                      'loss_psnr',
                                      'loss_fft'
                                     ]))


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizers, schdulers, metrics, model_dir,
                       restore_file=None):
    # reload weights from restore_file if specified

    # if restore_file is not None:
    #     restore_path = os.path.join(save_path, args.trained_para)
    #     logging.info("Restoring parameters from {}".format(restore_path))
    #     utils.load_checkpoint(restore_path, model, optimizers)

    best_val_acc = 0.0

    '''
    # train add logger,epoch two parameters
    '''

    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(opt.start_epoch,opt.epochs):
        # Run one epoch
        #         scheduler.step()
        logging.info("Epoch {}/{}".format(epoch + 1, opt.epochs))
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizers, schedulers, train_dataloader,epoch, model_dir,
                           restore_file)
        # Evaluate for one epoch on validation set
        eval_loss_savefilename = opt.loss_save_path + '/eval_LOSS_%04d.txt' % (epoch)
        evaluate(model, val_dl, eval_loss_savefilename,epoch,viz_1)

        # Save weights
        # utils.save_checkpoint({'epoch': epoch + 1,
        #                        'state_dict': model.state_dict(),
        #                        'optim_dict': optimizer.state_dict()},
        #                       is_best=is_best,
        #                       checkpoint=model_dir)

        # If best_eval, best_save_path
            # Save best val metrics in a json file in the model directory
        #     best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
        #     utils.save_dict_to_json(val_metrics, best_json_path)
        #
        # # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)

    plt.plot(global_loss)
    plt.savefig("final loss.jpg")

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
    train_dataloaders = fetch_dataloader(['train'], args.label_data_dir,args.input_t_length,opt.batch_size,
                                         downsample=args.downsample_form,downsample_index=args.upsample_index,time_interp=args.time_interp)
    eval_dataloaders = fetch_dataloader(['val'], args.eval_label_data_dir, args.input_t_length, opt.batch_size,
                                         downsample=args.downsample_form,downsample_index=args.upsample_index,time_interp=args.time_interp)

    train_dl = train_dataloaders['train']
    val_dl = eval_dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer

    #model = get_paras().to(device)  #
    #model=SRCNN(num_channels=2).to(device)
    #model = SRCNN(num_channels=2).to(device)
    # model = SRCNN(scale=8).to(device)
    model =torch.nn.ModuleList([RFDN(upscale=args.upsample_index, start_iter_num=args.start_iter_num,NS_ECA=args.NS_ECA).to(device),
                                RFDN(upscale=args.upsample_index, start_iter_num=args.start_iter_num,NS_ECA=args.NS_ECA).to(device),
                                RFDN(upscale=args.upsample_index, start_iter_num=args.start_iter_num,NS_ECA=args.NS_ECA).to(device),


])
    if args.use_bayes:
        bayes_nn=BayesNN(model, n_samples=len(model)  , noise=opt.var_eq).to(device)
        if args.trained_para is not None:
            data = torch.load(args.trained_para,map_location=torch.device('cpu'))
            model_dict = bayes_nn.state_dict()
            state_dict = {}
            for k, v in data.items():
                if k in model_dict.keys():
                    if args.upsample_index==4:
                        if k=='nnets.0.upsampler2.0.weight':
                            continue
                        elif k=='nnets.0.upsampler2.0.bias':
                            continue
                        elif k=='nnets.1.upsampler2.0.weight':
                            continue
                        elif k=='nnets.1.upsampler2.0.bias':
                            continue
                        elif k=='nnets.2.upsampler2.0.weight':
                            continue
                        elif k=='nnets.2.upsampler2.0.bias':
                            continue
                        else:
                            state_dict[k] = v
                    else:
                        state_dict[k] = v
            # state_dict={k:v for k,v in data.items() if k in model_dict.keys()}
            # state_dict['en.Subp.netMain.0.weight']=torch.randn([128,130,3,3],requires_grad=True,dtype=torch.float32)
            model_dict.update(state_dict)
            bayes_nn.load_state_dict(model_dict)
        svgd = SVGD(bayes_nn, train_dl)
        print('Start training.........................................................')
        tic = time.time()
        data_likelihod = []
        eq_likelihood = []
        rec_lamda = []
        rec_beta_eq = []
        rec_log_beta = []
        rec_log_alpha = []
        LOSS, LOSSB, LOSS1, LOSS2, LOSS3, LOSSD, LOSS4 = [], [], [], [], [], [], []
        metrics = {
            'PSNR': psnr,
            'SSIM': ssim,
            # could add more metrics such as accuracy for each token type
        }
        viz_1 = visdom.Visdom()
        for epoch in range(opt.start_epoch,opt.epochs):
            para_savefilename = save_path + '/finetune_%04d.pth'%(epoch)
            loss_savefilename = opt.loss_save_path + '/LOSS_%04d.pth' % (epoch)
            eval_loss_savefilename=opt.loss_save_path + '/eval_LOSS_%04d.txt' % (epoch)
            evaluate(bayes_nn, val_dl, eval_loss_savefilename, epoch, viz_1)
            bayes_nn_trained, data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq, rec_log_beta, rec_log_alpha, LOSS, LOSS1, LOSS2, LOSS3, LOSS4 = \
                svgd.train(epoch,data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq,
              rec_log_beta, rec_log_alpha,LOSS,LOSS1,LOSS2,LOSS3, LOSS4,para_savefilename,loss_savefilename,viz_1,args.upsample_index,args.time_interp)
            #evaluate(bayes_nn, val_dl, eval_loss_savefilename, epoch, viz_1)

    else:
        viz_1 = visdom.Visdom()
        model = BayesNN(model, n_samples=len(model), noise=opt.var_eq).to(device)
        if args.trained_para is not None:
            data = torch.load(args.trained_para)
            model_dict = model.state_dict()
            state_dict = {}
            for k, v in data.items():
                if k in model_dict.keys():
                    state_dict[k] = v
            # state_dict={k:v for k,v in data.items() if k in model_dict.keys()}
            # state_dict['en.Subp.netMain.0.weight']=torch.randn([128,130,3,3],requires_grad=True,dtype=torch.float32)
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        optimizers = []
        schedulers = []
        for i in range(3):
            parameters=[{'params':model.parameters()}]
            optimizer_i=torch.optim.AdamW(parameters, args.lr, weight_decay=.00005, eps=1e-8)
            optimizers.append(optimizer_i)
            MAX_STEP = 5
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_i, MAX_STEP, eta_min=args.lr*0.5))

        # fetch loss function and metrics
        loss_fn = loss_fn
        # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
        metrics = {
            'PSNR': psnr,
            'SSIM': ssim,
            # could add more metrics such as accuracy for each token type
        }

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(opt.epochs))
        train_and_evaluate(model, train_dl, val_dl, optimizers, schedulers, metrics, args.para_save_path,
                           args.loss_save_path)
        print("finish training and evaluating!")