import torch
import torch.nn as nn
import model_RFDN.block_3D as B
import time
import torch.nn.functional as F
from N_S_loss_Fan_init_coord import cal_NS_residual_vv_rotate
def make_model(args, parent=False):
    model = RFDN()
    return model
def grad_i_x(u,kernelx,x_step,pad=0):
    RepPad = nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
    u_x = F.conv3d(u, kernelx, bias=None,stride=1) / x_step
    if pad:
        u_x = RepPad(u_x)
    return u_x
def vel_to_vor_3d(u,v,w,x_step,y_step,z_step):
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
    kernely = kernelx.transpose(1, 2)
    kernelz = kernelx.transpose(2, 0)
    u_x=grad_i_x(u,kernelx,1,pad=1)
    u_y=grad_i_x(u,kernely,1,pad=1)
    u_z = grad_i_x(u, kernelz, 1,pad=1)

    v_x= grad_i_x(v,kernelx,1,pad=1)
    v_y= grad_i_x(v,kernely,1,pad=1)
    v_z = grad_i_x(v, kernelz, 1,pad=1)

    w_x=grad_i_x(w,kernelx,1,pad=1)
    w_y=grad_i_x(w,kernely,1,pad=1)
    w_z = grad_i_x(w, kernelz, 1,pad=1)

    x_diff_grid=grad_i_x(x_step,kernelx,1)
    y_diff_grid=grad_i_x(y_step,kernely,1)
    z_diff_grid=grad_i_x(z_step,kernelz,1)

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

    return vor_x,vor_y,vor_z

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv3d(hidden_dim+input_dim, hidden_dim, 3,stride=1, padding=1)
        self.convr = nn.Conv3d(hidden_dim+input_dim, hidden_dim, 3, stride=1,padding=1)
        self.convq = nn.Conv3d(hidden_dim+input_dim, hidden_dim, 3,stride=1, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h
def norm_input(input):
    mean=input.mean()
    return (input-mean)/mean, mean
def denorm_input(input,mean):
    return (input*mean)+mean

class RFDN(nn.Module):
    def __init__(self, in_nc=6, nf=48, num_modules=4, out_nc=6, upscale=2,dowm_sample=False,start_iter_num=1, NS_ECA=False): ##改
        super(RFDN, self).__init__()
        self.nf=nf
        self.NS_ECA=NS_ECA
        self.isdownsample=dowm_sample
        self.start_iter_num=start_iter_num
        self.fea_conv = B.conv_block(in_nc,nf,kernel_size=3,act_type='swish',groups=6)
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        #self.c_GRU = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='swish')
        self.c = B.conv_block(nf * num_modules, nf , kernel_size=1, act_type='swish')
        self.downsample=torch.nn.MaxPool3d(kernel_size=2,stride=2)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block
        if self.isdownsample:
            self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale*2)
        else:
            self.upsampler2 = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0
        self.GRU_block = ConvGRU(nf, nf)
        self.GRU_middle_block = ConvGRU(nf, nf)
        self.flow_head=B.conv_block(nf,nf,kernel_size=3,act_type='swish')
        self.flow_head_1 = B.conv_block(nf, nf, kernel_size=3, act_type='swish')
        self.B_middle_block=B.conv_block(nf* num_modules,nf* num_modules,kernel_size=3,act_type='swish')
        self.sigmoid = nn.Sigmoid()

    def show_middle_layer(self,input):
        input=input.detach().cpu().numpy()
        t,c,d,h,w=input.shape
        t=int(t/2)
        d = int(d / 2)
        h = int(h / 2)
        w = int(w / 2)
        input=input[t,0,d]
        import matplotlib.pyplot as plt
        plt.imshow(input,cmap='jet',vmax=1,vmin=-2)
        plt.colorbar()
        plt.show()
    def NS_ATT(self):
        u=self.input[:,0].unsqueeze(1)
        v=self.input[:,1].unsqueeze(1)
        w=self.input[:,2].unsqueeze(1)
        step_x=self.steps[0][:,:,::2,::2,::2]
        step_y=self.steps[1][:,:,::2,::2,::2]
        step_z=self.steps[2][:,:,::2,::2,::2]

        aresx,aresy,aresz,_=cal_NS_residual_vv_rotate(u, v, w,
         step_x, step_y,step_z, self.t_step, self.density, self.viscosity)
        NS_att=abs(aresx)+abs(aresy)+abs(aresz)

        #NS_att=(self.sigmoid(NS_att)-0.5)*2
        NS_att=self.sigmoid(NS_att)
        one_pad=torch.ones_like(NS_att)
        _,_,dn,hn,wn=NS_att.shape
        RepPad = nn.ReplicationPad3d(padding=(2, 2,2, 2, 2, 2))
        NS_att=RepPad(NS_att)
        zero_pad=torch.zeros_like(NS_att)
        zero_pad[:,:,2:2+dn,2:2+hn,2:2+wn]=one_pad
        #print(zero_pad)
        NS_att=NS_att*zero_pad
        #print(NS_att.max())
        zero_pad=0.5*(1-zero_pad)
        NS_att+=zero_pad
        #self.show_middle_layer(NS_att)
        self.show_middle_layer(NS_att)
        NS_att=NS_att.mean(dim=0)

        self.NS_att=NS_att



    def forward(self, input,steps=None, t_step=None, density=None, viscosity=None, t_interp=True):
        self.input=input
        self.show_middle_layer(self.input)
        self.steps=steps
        self.t_step=t_step
        self.density=density
        self.viscosity=viscosity
        if self.NS_ECA:
            self.NS_ATT()
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        if self.isdownsample:
            out_fea=self.downsample(out_fea)
            out_B1_down = self.downsample(out_B1)
        else:
            out_B1_down=out_B1
        if self.NS_ECA:
            out_B2 = self.B2(out_B1_down) * self.NS_att
        else:
            out_B2 = self.B2(out_B1_down)
        out_B2_down=out_B2
        if self.NS_ECA:
            out_B3 = self.B3(out_B2_down) * self.NS_att
        else:
            out_B3 = self.B3(out_B2_down)
        out_B3_down=out_B3
        if self.NS_ECA:
            out_B4 = self.B4(out_B3_down)*self.NS_att
        else:
            out_B4 = self.B4(out_B3_down)
        # self.show_middle_layer(out_B1)
        # self.show_middle_layer(out_B2)
        # self.show_middle_layer(out_B3)
        # self.show_middle_layer(out_B4)
        _,_,D,H,W=out_B1.shape
        out_B=torch.cat([out_B1_down,
                         out_B2_down,
                         out_B3_down,
                         out_B4], dim=1)
        out_B_GRU = self.c(out_B[0].unsqueeze(dim=0))
        net1 = torch.tanh(out_B_GRU)
        delta_flow = self.flow_head(out_B_GRU)
        #net_middle=self.c(0.5*(out_B[1].unsqueeze(dim=0)+out_B[0].unsqueeze(dim=0)))
        flow_pre=[delta_flow]
        t_length = out_B.shape[0]
        for t in range(1,t_length):
            out_B_GRU_middle = self.c(self.B_middle_block(0.5*(out_B[t].unsqueeze(dim=0)+out_B[t-1].unsqueeze(dim=0))))
            out_B_GRU=self.c(out_B[t].unsqueeze(dim=0))
            net2 = self.GRU_block(net1, out_B_GRU)
            net_middle1=  0.5*(net1+net2)
            net_middle2 = self.GRU_block(net_middle1, out_B_GRU_middle)
            if t >=self.start_iter_num:
                delta_flow = self.flow_head(net2)
                delta_flow_middle = self.flow_head(net_middle2)
                net1=net2
                if t_interp:
                    flow_pre.append(delta_flow_middle)
                    flow_pre.append(delta_flow)
                else:
                    flow_pre.append(delta_flow)
        out_fea_interp=[]
        for i in range(t_length-self.start_iter_num):
            middle_fea=out_fea[i]+out_fea[i+1]
            if t_interp:
                out_fea_interp.append(out_fea[i].unsqueeze(dim=0))
                out_fea_interp.append(middle_fea.unsqueeze(dim=0))
            else:
                out_fea_interp.append(out_fea[i].unsqueeze(dim=0))
        out_fea_interp.append(out_fea[-1].unsqueeze(dim=0))
        out_fea_interp=torch.cat(out_fea_interp,dim=0)
        flow_pre=torch.cat(flow_pre,dim=0)

        #out_lr =  out_fea_interp
        #out_lr = self.LR_conv(flow_pre)
        out_lr = self.LR_conv(flow_pre) + out_fea_interp

        #self.show_middle_layer(out_lr)
        #out_lr=out_lr.detach()
        if self.isdownsample:
            output = self.upsampler(out_lr)
        else:
            output = self.upsampler2(out_lr)
        torch.cuda.empty_cache()
        return output



if __name__ == '__main__':
    x = torch.rand(5,6,16,32,64).cuda()
    print(x.shape)
    model = RFDN(upscale=2).cuda()
    # flops, params = profile(model, inputs=(x, ))
    params = sum(param.nelement() for param in model.parameters())
    print(params / 1e6)
    t0=time.time()
    out = model(x,start_iter_num=2)
    print(out.shape)
