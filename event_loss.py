import torch
from torchvision.utils import save_image
import math
import random
import os

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()


def event_loss_call(all_rgb, event_data, combination, rgb2gray, resolution_h, resolution_w, iteration, img_i):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    #NeRFだから，ある座標select coordsについてやってる
    #E2NeRF, 32*32*4のランダムなピクセルについてやってる 
    loss = []
    chose = random.sample(combination, 10)
    for its in range(10):
        start = chose[its][0]
        end = chose[its][1]
        thres_pos = (lin_log(torch.sum(all_rgb[end] * rgb2gray, dim=0)*255) - lin_log(torch.sum(all_rgb[start] * rgb2gray, dim=0)*255)) / 0.3
        thres_neg = (lin_log(torch.sum(all_rgb[end] * rgb2gray, dim=0)*255) - lin_log(torch.sum(all_rgb[start] * rgb2gray, dim=0)*255)) / 0.2
        event_clone = event_data.clone()
        event_cur = event_clone[start].view(resolution_h, resolution_w)
        for j in range(start + 1, end):
            event_cur += event_clone[j].view(resolution_h, resolution_w)
        
        pos = event_cur >= 0
        neg = event_cur <= 0
        zero = (event_cur == 0)

        loss_pos = torch.mean(((thres_pos * pos) - ((event_cur + 0.5) * pos)) ** 2)
        loss_neg = torch.mean(((thres_neg * neg) - ((event_cur - 0.5) * neg)) ** 2)

        loss.append(loss_pos + loss_neg)

    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss