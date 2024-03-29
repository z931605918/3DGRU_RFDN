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

## Import local modules (Prof.JX-W's python code)
RWOF_dir = os.path.expanduser("/home/luning/Documents/utility/pythonLib")
RWOF_dir_1 = os.path.expanduser("/home/luning/Documents/utility/pythonLib/python_openFoam")
sys.path.append(RWOF_dir)
sys.path.append(RWOF_dir_1)
import subprocess # Call the command line
from subprocess import call
# import the modules you need
#import foamFileOperation as foamOp
from FCN import Net
from BayesNN import BayesNN




# svgd
# helpers
#log_sum_exp, parameters_to_vector, vector_to_parameters,_check_param_device
def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable LogSumExp.
    Args:
        input (Tensor)
        dim (int): Dimension along with the sum is performed
        keepdim (bool): Whether to retain the last dimension on summing
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    max_val = inputs.max(dim=dim, keepdim=True)[0]
    output = max_val + (inputs - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output

def parameters_to_vector(parameters, grad=False, both=False):
    """Convert parameters or/and their gradients to one vector
    Arguments:
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
        grad (bool): Vectorizes gradients if true, otherwise vectorizes params
        both (bool): If True, vectorizes both parameters and their gradients,
            `grad` has no effect in this case. Otherwise vectorizes parameters
            or gradients according to `grad`.
    Returns:
        The parameters or/and their gradients (each) represented by a single
        vector (th.Tensor, not Variable)
    """
    # Flag for the device where the parameter is located
    param_device = None

    if not both:
        vec = []
        if not grad:
            for param in parameters:
                # Ensure the parameters are located in the same device
                param_device = _check_param_device(param, param_device)
                vec.append(param.data.view(-1))
        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.data.view(-1))
        return torch.cat(vec)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)

def vector_to_parameters(vec, parameters, grad=True):
    """Convert one vector to the parameters or gradients of the parameters
    Arguments:
        vec (torch.Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
        grad (bool): True for assigning de-vectorized `vec` to gradients
    """
    '''
    # Ensure vec of type Variable
    if not isinstance(vec, torch.cuda.FloatTensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    '''
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    if grad:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(
                param.size())
            # Increment the pointer
            pointer += num_param
    else:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[pointer:pointer + num_param].view(
                param.size())
            # Increment the pointer
            pointer += num_param

def _check_param_device(param, old_param_device):
    """This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Variable]): a Variable of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device