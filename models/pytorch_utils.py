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

def forward(model, x, batch_size):
    """Forward data to model in mini-batch. 
    
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    total_segments = int(np.ceil(len(x) / batch_size))
    
    while True:
        print('Segment {} / {}'.format(pointer, total_segments))
        if pointer >= len(x):
            break

        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)

        for key in batch_output_dict.keys():
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
