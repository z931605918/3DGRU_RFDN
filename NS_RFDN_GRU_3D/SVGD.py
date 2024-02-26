import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
import copy
import math
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
from time import time
import time as t
import sys
import os
import gc
from visdom import Visdom

from helpers import log_sum_exp, parameters_to_vector, vector_to_parameters, _check_param_device
from args import args, device

n_samples = args.n_samples
lr = args.lr
lr_noise = args.lr_noise
ntrain = args.ntrain
class SVGD(object):
    """
    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, bayes_nn,train_loader):
        """
        For-loop implementation of SVGD.
        Args:
            bayes_nn (nn.Module): Bayesian NN
            train_loader (utils.data.DataLoader): Training data loader
            logger (dict)
        """
        self.bayes_nn = bayes_nn
        self.train_loader = train_loader
        self.n_samples = n_samples
        self.optimizers, self.schedulers = self._optimizers_schedulers(
                                            lr, lr_noise)
    def _squared_dist(self, X):
        """Computes squared distance between each row of `X`, ||X_i - X_j||^2
        Args:
            X (Tensor): (S, P) where S is number of samples, P is the dim of
                one sample
        Returns:
            (Tensor) (S, S)
        """
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)



    def _Kxx_dxKxx(self, X):

        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.
        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self._squared_dist(X)
        #squared_dist.median()中位数
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx


        '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    '''
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        print('shape of theta is',self.theta.shape)
        print('shape of sq_dist is',sq_dist.shape)
        print('shape of pairwise_dists is',pairwise_dists.shape)
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel

        Kxy = np.exp( -pairwise_dists / h**2 / 2)
        print('Kxy.shape is',Kxy.shape)
        time.sleep(1)
        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        print('dxkxy.shape is',dxkxy.shape)
        time.sleep(1)
        return (Kxy, dxkxy)

    def loss_fun(self,output,target):
        result_by = (torch.mean(abs(output - target)))
        result_by =torch.sqrt((output - target).pow(2).mean())
        # out=output.detach().cpu().numpy()
        # out_1=np.mean(np.mean(out,axis=4),axis=3)
        #result_by = (torch.mean((output - target) ** 2) ).sqrt()
        return result_by

    def _optimizers_schedulers(self, lr, lr_noise):
        """Initialize Adam optimizers and schedulers (ReduceLROnPlateau)
        Args:
            lr (float): learning rate for NN parameters `w`
            lr_noise (float): learning rate for noise precision `log_beta`
        """
        optimizers = []
        schedulers = []
        for i in range(self.n_samples):
            # parameters = [{'params': [self.bayes_nn[i].log_beta],'lr':lr_noise},
            #         {'params': self.bayes_nn[i].parameters()}]
            parameters = [{'params': self.bayes_nn[i].parameters()}]

            #optimizer_i = torch.optim.Adam(parameters, lr=lr)
            #optimizer_i=torch.optim.SGD(parameters,lr=lr)

            optimizer_i = torch.optim.AdamW(parameters, lr, weight_decay=.00005, eps=1e-8)
            optimizers.append(optimizer_i)
            # schedulers.append(ReduceLROnPlateau(optimizer_i,
            #         mode='min', factor=0.8, patience=50, verbose=True))
            MAX_STEP = 5
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_i, MAX_STEP, eta_min=lr*0.5))
        return optimizers, schedulers

    def train(self, epoch,data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq,
              rec_log_beta, rec_log_alpha,LOSS,LOSS1,LOSS2,
              LOSS3,LOSS4,savefilename,lossfilename,viz_1,upsample_index,time_interp):
        # python -m visdom.server
        self.bayes_nn.train()
        rmse_show=0.
        loss_1_show=0.
        loss_2_show = 0.
        loss_3_show = 0.
        loss_4_show = 0.
        loss_m_show = 0.
        loss_fft_show=0.

        for batch_idx, (input_dict, target_dict) in enumerate(self.train_loader):
            if upsample_index==2:
                steps=(target_dict['x_grid'][0].to(device).float(),
                       target_dict['y_grid'][0].to(device).float(),
                       target_dict['z_grid'][0].to(device).float())
                steps_down=steps
            elif upsample_index==4:
                steps = (target_dict['x_grid'][0].to(device), target_dict['y_grid'][0].to(device),
                         target_dict['z_grid'][0].to(device))
                kernel = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
                steps_down = (
                    kernel(target_dict['x_grid'][0].to(device)),
                    kernel(
                        target_dict['y_grid'][0].to(device)),
                    kernel(
                        target_dict['z_grid'][0].to(device))
                )
            density=target_dict['density'][0]
            viscosity=target_dict['viscosity'][0].cuda().float()
            t_step=target_dict['t_step'][0].cuda().float()
            input=input_dict['v_oumiga'].to(device)[0].float()
            target=target_dict['v_oumiga'].to(device)[0].float()
            self.bayes_nn.zero_grad()
            # all gradients of log joint probability: (S, P)
            grad_log_joint = []
            # all model parameters (particles): (S, P)
            theta = []
            theta_len=[]
            # store the joint probabilities
            log_joint = 0.
            loss_1, loss_2, loss_3, loss_4 = 0, 0, 0, 0
            t,C,Hd,Hw,Hh=target.shape
            output = torch.zeros_like(target)
            output_is=[]
            for i in range(self.n_samples):
                #####################
                ###modified for sparse data stenosis
                ## forward for training data
                self.optimizers[i].zero_grad()
                output_i = self.bayes_nn[i].forward(input,steps_down, t_step, density, viscosity,t_interp=time_interp)
                output_is.append(output_i)
                ot,oc,od,oh,oh=output_i.shape
                #output_i=torch.nn.functional.interpolate(output_i,(Hd,Hw,Hh),mode='trilinear',align_corners=False)
                output += output_i.detach()
                ## loss for unlabelled points TODO 计算物理损失，用auto_grad,grad计算导数
                log_eq_i,loss_1i,loss_2i,loss_3i,loss_4i,res_x_out,res_x_tag=self.bayes_nn.criterion_grid(output_i, target, steps, t_step, density, viscosity, i,ntrain)
                input_up=torch.nn.functional.interpolate(input,(Hd,Hw,Hh),mode='trilinear',align_corners=True)
                input_up_1=(input_up[0]+input_up[1])/2
                input_up_3=(input_up[1]+input_up[2])/2
                input_up=torch.stack([input_up[0],input_up_1,input_up[1],input_up_3,input_up[2]])
                _,res_x_input_sub,_,_,_,res_x_input,_=self.bayes_nn.criterion_grid(input_up, target, steps, t_step, density, viscosity, i,ntrain)

                ### for monity purpose  TODO 保存代码
                                ## loss for labelled points  计算likelihood TODO 和2016中一样
                log_joint_i = self.bayes_nn._log_joint(i, output_i, target, ntrain)
                loss_1, loss_2, loss_3, loss_4 = self._monitor(loss_1, loss_2, loss_3,
                                                               loss_4, loss_1i.detach(), loss_2i.detach(),
                                                               loss_3i.detach(), loss_4i.detach())
                ###总损失梯度反向传递
                log_eq_i=log_eq_i*0.002
                log_joint_i += log_eq_i
                log_joint_i=1e-5*log_joint_i
                log_joint_i.backward()
                if (i==0) :
                    rec_log_beta.append(self.bayes_nn[i].log_beta.item())
                    rec_beta_eq.append(self.bayes_nn[i].beta_eq)
                #####
                # backward frees memory for computation graph
                # computation below does not build computation graph
                # extract parameters and their gradients out from models
                # Todo 与2016一致
                vec_param, vec_grad_log_joint = parameters_to_vector(
                    self.bayes_nn[i].parameters(), both=True)
                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                theta.append(vec_param.unsqueeze(0))
                theta_len.append(int(vec_param.shape[0]))
            max_theta_len=np.array(theta_len).max()
            theta_pad=[]
            grad_log_joint_pad=[]

            # calculating the kernel matrix and its gradients
            # Todo 与2016一致
            for theta_i, theta_i_len in zip(theta,theta_len):
                len_below=max_theta_len-theta_i_len
                zeros_pad=torch.zeros((1,len_below)).to(device)
                theta_i=torch.cat([theta_i,zeros_pad],dim=1)
                theta_pad.append(theta_i)
            for grad_log_joint_i, theta_i_len in zip(grad_log_joint,theta_len):
                len_below=max_theta_len-theta_i_len
                zeros_pad=torch.zeros((1,len_below)).to(device)
                grad_log_joint_i=torch.cat([grad_log_joint_i,zeros_pad],dim=1)
                grad_log_joint_pad.append(grad_log_joint_i)
            theta = torch.cat(theta_pad)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint_pad)
            grad_logp = torch.mm(Kxx, grad_log_joint)
            # negate grads here!!!
            grad_theta = - (grad_logp + dxKxx) / self.n_samples
            ## switch back to 1 particle
            #grad_theta = grad_log_joint
            # explicitly deleting variables does not release memory :(
            output=output/self.n_samples
            loss_rmse,mean= self.normal_rmse(output,target)
            rmse=loss_rmse/mean
            rmse_show+=rmse
            input_show=nn.functional.interpolate(input,size=(Hd,Hw,Hh),mode='trilinear',align_corners=False)

            target_np=target.detach().cpu().numpy()

            output_fft = torch.log10(abs(torch.fft.fftn(output[:, :3].cpu(), norm='forward')))  # .cpu().detach().numpy()
            target_fft = torch.log10(abs(torch.fft.fftn(target[:, :3].cpu(), norm='forward') ))  # .cpu().detach().numpy()

            target_fft_cpu = torch.log10(abs(torch.fft.fftn(target.cpu(), norm='forward')))
            fft_rmse,fft_mean=self.normal_rmse(output_fft,target_fft)
            fft_rmse=fft_rmse/fft_mean
            loss_fft_show+=fft_rmse

            output_fft=output_fft.detach().cpu().numpy()
            target_fft=target_fft.detach().cpu().numpy()
            target_fft_cpu=target_fft_cpu.detach().cpu().numpy()

            targetfft_np = np.log10(abs(np.fft.fftn(target_np) / target_np.size))

            fft_max=float(target_fft.max())
            fft_min=float(target_fft.min())
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



            # update param gradients
            # WEAK: no loss function to suggest when to stop or
            # approximation performance
            #mse = F.mse_loss(output / self.n_samples, target).item()
            loss_1/= self.n_samples
            loss_2/=self.n_samples
            loss_3/=self.n_samples
            loss_4/=self.n_samples
            loss = (loss_1.cpu().numpy() + loss_2.cpu().numpy() + loss_3.cpu().numpy() + loss_4.cpu().numpy()) / 4
            loss_1_show += loss_1.cpu().numpy()
            loss_2_show += loss_2.cpu().numpy()
            loss_3_show += loss_3.cpu().numpy()
            loss_4_show += loss_4.cpu().numpy()
            loss_m_show+=loss
            LOSS.append(loss)
            LOSS1.append(loss_1.cpu().numpy())
            LOSS2.append(loss_2.cpu().numpy())
            LOSS3.append(loss_3.cpu().numpy())
            LOSS4.append(loss_4.cpu().numpy())
            if batch_idx %2 ==0:
                viz_1.heatmap(X=target[1,0,9].cpu().numpy(), win='ulabel',
                              opts=dict(title='ulabel', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=output[1,0,9].cpu().numpy(), win='uout',
                              opts=dict(title='uout1', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=target[1,1,9].cpu().numpy(), win='vlabel',
                              opts=dict(title='vlabel', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=output[1,1,9].cpu().numpy(), win='vout',
                              opts=dict(title='vout1', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=target[1,2,9].cpu().numpy(), win='wlabel',
                              opts=dict(title='wlabel', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=output[1,2,9].cpu().numpy(), win='wout',
                              opts=dict(title='wout1', colormap='rainbow', xmax=1.5, xmin=-1))

                viz_1.heatmap(X=input_show[1,0,9].cpu().numpy(), win='uinput',
                              opts=dict(title='uinput', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=input_show[1,1,9].cpu().numpy(), win='vinput',
                              opts=dict(title='vinput', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=input_show[1,2,9].cpu().numpy(), win='winput',
                              opts=dict(title='winput', colormap='rainbow', xmax=1.5, xmin=-1))

                viz_1.heatmap(X=input_show[1,0,9].cpu().numpy()-target[1,0,9].cpu().numpy(), win='uinput-res',
                              opts=dict(title='uinput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
                viz_1.heatmap(X=input_show[1,1,9].cpu().numpy()-target[1,1,9].cpu().numpy(), win='vinput-res',
                              opts=dict(title='vinput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
                viz_1.heatmap(X=input_show[1,2,9].cpu().numpy()-target[1,2,9].cpu().numpy(), win='winput-res',
                              opts=dict(title='winput残差', colormap='rainbow', xmax=0.1, xmin=-0.1))

                viz_1.heatmap(X=target[1,0,9].cpu().numpy()-output[1,0,9].cpu().numpy(), win='ures',
                              opts=dict(title='u残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
                viz_1.heatmap(X=target[1,1,9].cpu().numpy()-output[1,1,9].cpu().numpy(), win='vres',
                              opts=dict(title='v残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
                viz_1.heatmap(X=target[1,2,9].cpu().numpy()-output[1,2,9].cpu().numpy(), win='wres',
                              opts=dict(title='w残差', colormap='rainbow', xmax=0.1, xmin=-0.1))
                viz_1.heatmap(X=loss_1i[0,0,9].detach().cpu().numpy(), win='xres',
                              opts=dict(title='Ns_x残差相减', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=loss_2i[0,0,9].detach().cpu().numpy(), win='yres',
                              opts=dict(title='Ns_y残差相减', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=loss_3i[0,0,9].detach().cpu().numpy(), win='zres',
                              opts=dict(title='Ns_z残差相减', colormap='rainbow', xmax=1.5, xmin=-1))

                viz_1.heatmap(X=res_x_out[0, 0, 9].detach().cpu().numpy(), win='xres_out',
                              opts=dict(title='Ns_x_out_残差', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=res_x_tag[0, 0, 9].detach().cpu().numpy(), win='xres_tag',
                              opts=dict(title='Ns_x_target_残差', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=res_x_input[0, 0, 9].detach().cpu().numpy(), win='xres_in',
                              opts=dict(title='Ns_x_input_残差', colormap='rainbow', xmax=1.5, xmin=-1))
                viz_1.heatmap(X=res_x_input_sub[0, 0, 9].detach().cpu().numpy(), win='xres_in_sub',
                              opts=dict(title='Ns_x_input_相减', colormap='rainbow', xmax=1.5, xmin=-1))

                viz_1.heatmap(X=output_fft[0, 0, :, 5], win='out_fft',
                              opts=dict(title='out_fft', colormap='rainbow',xmax=fft_max,xmin=fft_min))
                viz_1.heatmap(X=target_fft[0, 0, :, 5], win='targte_fft',
                              opts=dict(title='target_fft', colormap='rainbow',xmax=fft_max,xmin=fft_min))
                viz_1.heatmap(X=output_fft[0, 0, :, 5]-target_fft[0, 0, :, 5], win='fft_sub',
                              opts=dict(title='fft相减', colormap='rainbow'))

            #print('len(self.train_loader)',len(self.train_loader))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg_N-S_Loss: {:.6f} Normal RMSE_loss: {:.5f}, FFT norm loss={:.5f}, learning_rate*1e5={:.2f}'.format(
                epoch, batch_idx, len(self.train_loader),
                100. * batch_idx / len(self.train_loader), loss, rmse,fft_rmse,
                1e5*self.schedulers[i].optimizer.param_groups[0]['lr']))
            if rmse / (batch_idx + 1) < 5:
                for i in range(self.n_samples):
                    vector_to_parameters(grad_theta[i, :theta_len[i]],
                                         self.bayes_nn[i].parameters(), grad=True)
                    self.optimizers[i].step()
            else:
                for i in range(self.n_samples):
                    self.bayes_nn[i].zero_grad()

            if batch_idx % 20 == 0:
                iter_num = batch_idx + (epoch) * len(self.train_loader)
                viz_1.line([[loss_m_show/ (batch_idx + 1),
                             loss_1_show/ (batch_idx + 1),
                             loss_2_show/ (batch_idx + 1),
                             loss_3_show/ (batch_idx + 1),
                             loss_4_show/ (batch_idx + 1),
                             loss_fft_show/(batch_idx+1),
                             rmse_show/ (batch_idx + 1) ]],[iter_num],win='training_line',update='append',
                                    opts=dict(title='training_line',
                                    legend=['mean_NS_loss',
                                            'NS_loss_X',
                                            'NS_loss_y',
                                            'NS_loss_z',
                                            'NS_loss_cont',
                                            'fft_loss','rmse'

                ]))

        rmse_train = rmse_show / len(self.train_loader)
        mean_LOSS1=  np.array(LOSS1).mean()
        mean_LOSS2 = np.array(LOSS2).mean()
        mean_LOSS3 = np.array(LOSS3).mean()
        mean_LOSS4 = np.array(LOSS4).mean()
        LOSSes={}
        LOSSes['rmse']=rmse_train
        LOSSes['LOSS1']=mean_LOSS1
        LOSSes['LOSS2'] = mean_LOSS2
        LOSSes['LOSS3'] = mean_LOSS3
        LOSSes['LOSS4'] = mean_LOSS4
        torch.save(LOSSes, lossfilename)
        # viz_1.line([[mean_LOSS1,
        #              mean_LOSS2,
        #              mean_LOSS3,
        #              mean_LOSS4,
        #              rmse_train,
        # ]],
        #            [epoch], win='training_line_epoch', update='append', opts=dict(title='training_line',
        #                      legend=[
        #                              'NS_loss_X',
        #                              'NS_loss_y',
        #                              'NS_loss_z',
        #                              'NS_loss_cont',
        #                              'rmse'
        #                              ]))
        #self._savept(rec_log_beta, rec_beta_eq, rec_log_alpha, mean_LOSS1,mean_LOSS2,mean_LOSS3,mean_LOSS4,rmse_train)
        viz_1.line([[loss_m_show / (batch_idx + 1),
                     loss_1_show / (batch_idx + 1),
                     loss_2_show / (batch_idx + 1),
                     loss_3_show / (batch_idx + 1),
                     loss_4_show / (batch_idx + 1),
                     rmse_train ,
                     loss_fft_show/(batch_idx + 1)]], [epoch], win='training_line_epoch', update='append',
                   opts=dict(title='training_line_epoch',
                             legend=['mean_NS_loss',
                                     'NS_loss_X',
                                     'NS_loss_y',
                                     'NS_loss_z',
                                     'NS_loss_cont',
                                     'rmse',
                                     'fft_rmse'
                                     ]))
        if epoch%1==0:
            self.bayes_nn.eval()
            torch.save(self.bayes_nn.state_dict(), savefilename)
        for i in range(self.n_samples):
            self.schedulers[i].step(epoch)
        return self.bayes_nn ,data_likelihod, eq_likelihood, rec_lamda, rec_beta_eq, rec_log_beta, rec_log_alpha, LOSS,LOSS1,LOSS2,LOSS3, LOSS4
        #np.savetxt('.csv',rec_lamda)


    def _plot(self,x,y,u):
        plt.figure()
        plt.scatter(x,y,c = u)
        plt.colorbar()
        plt.show()
    def _dataset(self,Data):
        sp_xin = Data['xinlet']
        sp_yin = Data['yinlet']
        sp_uin = Data['uinlet']
        sp_xout = Data['xoutlet']
        sp_yout = Data['youtlet']
        sp_uout = Data['uoutlet']
        sp_xdom = Data['xdom']
        #print('x_size is',sparse_x.shape)
        sp_ydom = Data['ydom']
        sp_udom = Data['udom']
        sp_vdom = Data['vdom']
        sp_Pdom = Data['pdom']
        sp_xb = Data['xb']
        sp_yb = Data['yb']
        sp_ub = Data['ub']
        sp_vb = Data['vb']
        sp_Pb = Data['pb']
        return sp_xin, sp_yin, sp_uin, sp_xout, sp_yout, sp_uout, sp_xdom, sp_ydom, sp_udom, sp_vdom, sp_Pdom, sp_xb, sp_yb, sp_ub, sp_vb, sp_Pb
    def _bound_sample(self, x_left, x_right, x_up, x_down, y_left, y_right, y_up, y_down, ratio = 4, device = device):
        perm_x_left = np.random.randint(len(x_left), size=args.bound_sample)
        perm_y_left = perm_x_left

        x_l_sample =torch.Tensor(x_left[perm_x_left]).to(device)
        y_l_sample = torch.Tensor(y_left[perm_y_left]).to(device)
        # right boudanry sample
        perm_x_right = np.random.randint(len(x_right), size=args.bound_sample)
        perm_y_right = perm_x_right

        x_r_sample = torch.Tensor(x_right[perm_x_right]).to(device)
        y_r_sample = torch.Tensor(y_right[perm_y_right]).to(device)

        # up boundary sample
        perm_x_up = np.random.randint(len(x_up), size=ratio*args.bound_sample)
        perm_y_up = perm_x_up
        x_u_sample = torch.Tensor(x_up[perm_x_up]).to(device)
        y_u_sample = torch.Tensor(y_up[perm_y_up]).to(device)


        # low boundary sample
        perm_x_down = np.random.randint(len(x_down), size=ratio*args.bound_sample)
        perm_y_down = perm_x_down
        x_d_sample = torch.Tensor(x_down[perm_x_down]).to(device)
        y_d_sample = torch.Tensor(y_down[perm_y_down]).to(device)
        return x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample, y_r_sample, y_u_sample, y_d_sample
    def _addnoise(self,noise_lv, sp_udom, sp_vdom, sp_Pdom):
        for i in range(0,len(sp_udom)):
            u_error = np.random.normal(0, noise_lv*np.abs(sp_udom[i]), 1)

            v_error = np.random.normal(0, noise_lv*np.abs(sp_vdom[i]), 1)
            p_error = np.random.normal(0, noise_lv*np.abs(sp_Pdom[i]), 1)
            sp_udom[i] += u_error
            sp_vdom[i] += v_error
            sp_Pdom[i] += p_error

        return sp_udom, sp_vdom, sp_Pdom
    def _paste_b(self,x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample,y_r_sample,y_u_sample,y_d_sample, device = device):
        xb =torch.cat((x_l_sample,x_r_sample,x_u_sample,x_d_sample),0).to(device)
        yb = torch.cat((y_l_sample,y_r_sample,y_u_sample,y_d_sample),0).to(device)
        xb = xb.view(len(xb),-1)
        yb = yb.view(len(yb),-1)
        return xb, yb
    def _paste_d(self, sp_xdom, sp_xb, sp_ydom, sp_yb, sp_udom, sp_ub, sp_vdom, sp_vb, sp_Pdom, sp_Pb, device = device):
        ##
        sp_x = np.concatenate((sp_xdom,sp_xb),0)
        sp_y = np.concatenate((sp_ydom,sp_yb),0)
        sp_u = np.concatenate((sp_udom,sp_ub),0)
        sp_v = np.concatenate((sp_vdom,sp_vb),0)
        sp_P = np.concatenate((sp_Pdom,sp_Pb),0)
        sp_x, sp_y, sp_u, sp_v, sp_P = sp_x[...,None], sp_y[...,None], sp_u[...,None], sp_v[...,None], sp_P[...,None]
        sp_data = np.concatenate((sp_x,sp_y,sp_u,sp_v,sp_P),1)


        ##
        # for sparase stenosis
        sp_x, sp_y, sp_u, sp_v, sp_P = torch.Tensor(sp_data[:,0]).to(device), torch.Tensor(sp_data[:,1]).to(device), torch.Tensor(sp_data[:,2]).to(device), torch.Tensor(sp_data[:,3]).to(device), torch.Tensor(sp_data[:,4]).to(device)
        sp_x, sp_y, sp_u, sp_v, sp_P = sp_x.view(len(sp_x), -1), sp_y.view(len(sp_y), -1), sp_u.view(len(sp_u), -1), sp_v.view(len(sp_v), -1), sp_P.view(len(sp_P),-1)
        return sp_x, sp_y, sp_u, sp_v, sp_P
    def _paste_in(self,x_in, y_in, u_in,device = device):
        x_in = torch.Tensor(x_in).to(device)
        y_in = torch.Tensor(y_in).to(device)
        u_in = torch.Tensor(u_in).to(device)
        x_in, y_in, u_in = x_in.view(len(x_in),-1), y_in.view(len(y_in),-1), u_in.view(len(u_in),-1)
        return x_in, y_in, u_in

    def _monitor(self, loss_1, loss_2, loss_3, loss_4, loss_1i, loss_2i, loss_3i, loss_4i):
        #loss_f = nn.MSELoss()
        loss_1 += self.loss_fun(loss_1i,torch.zeros_like(loss_1i))
        loss_2 += self.loss_fun(loss_2i,torch.zeros_like(loss_2i))
        loss_3 += self.loss_fun(loss_3i,torch.zeros_like(loss_3i))
        loss_4 += self.loss_fun(loss_4i, torch.zeros_like(loss_4i))
        return loss_1, loss_2, loss_3, loss_4
    def _savept(self, rec_log_beta, rec_beta_eq, rec_log_alpha,mean_LOSS1,mean_LOSS2,mean_LOSS3,mean_LOSS4,rmse_train):
        np.savetxt('LOSS\\LOSS1.csv',mean_LOSS1)
        np.savetxt('LOSS\\LOSS2.csv', mean_LOSS2)
        np.savetxt('LOSS\\LOSS3.csv', mean_LOSS3)
        np.savetxt('LOSS\\LOSS4.csv', mean_LOSS4)
        np.savetxt('LOSS\\MSE.csv', rmse_train)
        np.savetxt('log_beta.csv',rec_log_beta)
        np.savetxt('beta_eq.csv',rec_beta_eq)
        np.savetxt('log_alpha.csv',rec_log_alpha)

    def normal_rmse(self,output, label):
        loss_rmse = torch.sqrt(((output[:, 0] - label[:, 0]).pow(2) + (output[:, 1] - label[:, 1]).pow(2) + (
                    output[:, 2] - label[:, 2]).pow(2)).mean()).cpu().numpy()
        mean = torch.sqrt((label[:, 0].pow(2) + label[:, 1].pow(2) + label[:, 2].pow(2)).mean()).cpu().numpy()
        return loss_rmse, mean
