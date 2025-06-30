import os
import numpy as np
import time
import torch

def move_data_to_device(x, device):
    """
        Moves data to device

        Args:
        -----
            x (torch.Tensor | np.ndarray)
            device (str): typical cuda or cpu

        Return:
        -------
            x (torch.Tensor): Tensor moved to device
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    """
        Append info to the dict

        Args:
        -----
            dict (dict): Original dictionary
            key (str): Key to insert
            value (unknown): Value to insert

        Returns:
        --------
            None
    """
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
