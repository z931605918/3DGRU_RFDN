import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
from PIL import Image
import matplotlib as mpl

import os, io, time, re
from glob import glob
from typing import Union, List, Tuple, Optional




TAG_STRING = 'PIEH'
TAG_FLOAT = 202021.25
flags = {
    'debug': False
}

"""
flowIO.h
"""
UNKNOWN_FLOW_THRESH = 1e9
def array_cropper(array, crop_window: Union[int, Tuple[int, int, int, int]] = 0):
    # Cropper init.
    s = array.shape  # Create image cropper
    crop_window = (crop_window,) * 4 if type(crop_window) is int else crop_window
    assert len(crop_window) == 4

    # Cropping the array
    return array[crop_window[0] : s[0]-crop_window[1], crop_window[2] : s[1]-crop_window[3]]
def read_flow(filename: str, crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> np.array: #
    """
        Read a .flo file (Middlebury format).
        Parameters
        ----------
        filename : str
            Filename where the flow will be read. Must have extension .flo.
        Returns
        -------
        flow : ndarray, shape (height, width, 2), dtype float32
            The read flow from the input file.
        """

    if not isinstance(filename, io.BufferedReader):
        if not isinstance(filename, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=filename))
        if not os.path.isfile(filename):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=filename))
        if not filename.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=filename.split('.')[-1]))

        flo = open(filename, 'rb')
    else:
        flo = filename

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    n_bands = 2
    size = n_bands * width * height
    tmp = np.frombuffer(flo.read(n_bands * width * height * 4), np.float32, count=size)
    flow = np.resize(tmp, (int(height), int(width), int(n_bands)))
    flo.close()

    return array_cropper(flow, crop_window=crop_window)
def write_flow(flow: np.ndarray, filename: str):
    """
    Write a .flo file (Middlebury format).
    Parameters
    ----------
    flow : ndarray, shape (height, width, 2), dtype float32
        Flow to save to file.
    filename : str
        Filename where flow will be saved. Must have extension .flo.
    norm : bool
        Logical option to normalize the input flow or not.
    Returns
    -------
    None
    """

    assert type(filename) is str, "file is not str (%r)" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo (%r)" % filename[-4:]

    height, width, n_bands = flow.shape
    assert n_bands == 2, "Number of bands = %r != 2" % n_bands

    # Extract the u and v velocity


    u = flow[:, :, 0]
    v = flow[:, :, 1]
    w = flow[:, :, 2] if flow.shape[2] > 2 else None

    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    with open(filename, 'wb') as f:
        TAG_FLOAT = 202021.25
        tag = np.array([TAG_FLOAT], dtype=np.float32)  # assign ASCCII tag for float 202021.25 (TAG_FLOAT)
        tag.tofile(f)
        np.array([width], dtype=np.int32).astype(np.int32).tofile(f)  # assign width size to ASCII
        np.array([height], dtype=np.int32).tofile(f)  # assign height size to ASCII
        flow.tofile(f)  # assign the array value (u, v)



