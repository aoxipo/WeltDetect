import cv2
from yolov5.utils.general import scale_boxes as scale_coords
from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model

def ai_init( weights = "/data/lijunlin/project/yolov5-master/runs/train/exp5/weights/best.pt", device = "cuda:0"):
    w = str(weights[0] if isinstance(weights, list) else weights)
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, device=device)   #加载模型
    return model


def warpper_img( img):
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = np.dstack( ( img, img, img)) 
    img_wappered = np.transpose(img, (2, 0, 1))
    img_wappered = ( img_wappered - img_wappered.min())/( img_wappered.max() - img_wappered.min() ) 
    return img_wappered

def inference(model, img0, device):
    img = np.expand_dims(img0, axis=0)    #扩展维度至[1,3,1024,1024]
    img = torch.from_numpy(img.copy())   #numpy转tensor
    img = img.to(torch.float32)          #float64转换float32
    img = img.to(device)
    pred = model(img, augment='store_true', visualize='store_true')[0]
    pred = pred.clone().detach()
     
    pred = non_max_suppression(pred, 0.35, 0.45, None, False, max_det=1000)  #非极大值抑制
    # pred的长度为1 pred[0].shape 为 (256,6)  ( point number, x0,y0,x1,y1,conf, cls)
    result_l = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape[1:]).round()
            result_l.append(det.cpu().numpy())
    return result_l

def get_mask(pred, origin_shape):
    mask = np.zeros(origin_shape)
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                mask[ int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = 1
    return mask
    

def show_result(pred, img, img0):
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # print('{},{},{}'.format(xyxy, conf, cls)) #输出结果：xyxy检测框左上角和右下角坐标，conf置信度，cls分类结果
                img0 = cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    # cv2.imwrite('out.jpg', img0)  #简单画个框
    return img0


if __name__ == "__main__":
    DEVICE = "cuda:0"
    WEIGHT_PATH = "/data/lijunlin/project/yolov5-master/runs/train/exp5/weights/best.pt"
    IMGAE_PATH = '/data/lijunlin/data/CT/PCB/images1/val/0067_12500.442_18014.914.jpg'

    img0 = cv2.imread(IMGAE_PATH, 0)
    img = warpper_img(img0)
    model = ai_init( WEIGHT_PATH, DEVICE)
    result = inference(model, img, DEVICE)
    mask = get_mask( result, img0.shape)
    rect_show = show_result(result, img, img0.copy())
