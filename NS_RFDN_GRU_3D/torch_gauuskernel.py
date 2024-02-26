# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import torch
import math
import torch.nn as nn


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_index=gaussian_kernel[1]
    gaussian_kernel_0=gaussian_kernel*gaussian_index[0]
    gaussian_kernel_1=gaussian_kernel*gaussian_index[1]
    gaussian_kernel_2=gaussian_kernel*gaussian_index[2]
    gaussian_kernel_3D=torch.stack([gaussian_kernel_0,gaussian_kernel_1,gaussian_kernel_2],dim=0)

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel_3D / torch.sum(gaussian_kernel_3D)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(1,channels, 1, 1,1)

    gaussian_filter = nn.Conv3d(in_channels=channels, out_channels=3, kernel_size=kernel_size,
                                bias=False, padding=kernel_size // 2,stride=2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
def get_mean_kernel(down_size):
    kernel=torch.nn.AvgPool3d(kernel_size=down_size,stride=down_size,padding=0)
    return kernel


# img = torch.randn([1,3,64,64,64]).cuda()
# blur_layer = get_gaussian_kernel().cuda()
#
# blured_img = blur_layer(img)
# print(blured_img.shape)
