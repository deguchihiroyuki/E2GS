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
        # print(all_rgb[end][0][0])
        # tensor_x = torch.sum(all_rgb[end] * rgb2gray, dim=0)
        # tensor_x = tensor_x * 255
        # # im = torch.permute(gt_image, (1,2,0))
        # gt_num = tensor_x.cpu().detach().numpy()
        # # gt_num = gt_num * 255
        # cv2.imwrite("G:\ssd1\gaussian-splatting\samplee\sample.png", gt_num)
        thres_pos = (lin_log(torch.sum(all_rgb[end] * rgb2gray, dim=0)*255) - lin_log(torch.sum(all_rgb[start] * rgb2gray, dim=0)*255)) / 0.3
        thres_neg = (lin_log(torch.sum(all_rgb[end] * rgb2gray, dim=0)*255) - lin_log(torch.sum(all_rgb[start] * rgb2gray, dim=0)*255)) / 0.2
        event_clone = event_data.clone()
        # print(thres_neg.size())
        # print(thres_pos.size())
        event_cur = event_clone[start].view(resolution_h, resolution_w)#ここを画像に投影して確認してみる,-1,1じゃない
        for j in range(start + 1, end):
            event_cur += event_clone[j].view(resolution_h, resolution_w)
        
        # if img_i == 0:
            # print(event_data.size())
            # print(torch.max(torch.abs(event_data)).item())
            # print(event_cur.size())
            # print(torch.max(torch.abs(event_cur)).item())
        #     max = torch.max(torch.abs(event_cur))
        #     x = 255//max.item()
        #     print(max.item())
        #     if x != 0:
        #         event_cur = torch.abs(event_cur)
        #         event_vis = event_cur * x
        #         save_path = r"G:\ssd1\gaussian-splatting\samplee\event_render"
        #         file_name = f"{iteration}.png"
        #         full_path = os.path.join(save_path, file_name)
        #         save_image(event_vis, full_path)
        # print(event_cur.size())#800*800各ピクセルのイベントの数
        # print(event_cur[0][0])
        pos = event_cur >= 0#0,1の800*800
        neg = event_cur <= 0
        zero = (event_cur == 0)

        loss_pos = torch.mean(((thres_pos * pos) - ((event_cur + 0.5) * pos)) ** 2)#(+-0.5がデフォルト)
        loss_neg = torch.mean(((thres_neg * neg) - ((event_cur - 0.5) * neg)) ** 2)
        # loss_zero = torch.mean((thres_pos * zero) ** 2) + torch.mean((thres_neg * zero) ** 2)
        # loss_zero = loss_zero * 5#イベントないところを重く計算してみる
        # if iteration >= 29000 and iteration % 100 == 0:
        #     print("pos_im", torch.mean(thres_pos * pos).item())
        #     print("pos_event", torch.mean((event_cur) * pos).item())
        #     print("neg_im", torch.mean(thres_neg * neg).item())
        #     print("neg_event", torch.mean((event_cur) * neg).item())
        #     print("loss_pos", loss_pos.item())
        #     print("loss_neg", loss_neg.item())
        loss.append(loss_pos + loss_neg)

    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss