import torch
import torch.nn as nn
import model_RFDN.block_3D as B
import time
def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=6, nf=48, num_modules=4, out_nc=6, upscale=2): ##改
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.downsample=torch.nn.MaxPool3d(kernel_size=2,stride=2)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


if __name__ == '__main__':
    x = torch.rand(5,6,16,32,64).cuda()
    print(x.shape)
    model = RFDN(upscale=2).cuda()
    # flops, params = profile(model, inputs=(x, ))
    params = sum(param.nelement() for param in model.parameters())
    print(params / 1e6)
    t0=time.time()
    out = model(x)
    print(out.shape)