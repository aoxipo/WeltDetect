import cv2
import numpy
import matplotlib.pyplot as plt
import os
import numpy as np
import configparser
%matplotlib inline
!pip install yolov5 -i https://pypi.tuna.tsinghua.edu.cn/simple

from ai_infer import *

def load_volume_from_ini(folder_path):
    # 1. 读取 ini 文件
    ini_path = os.path.join(folder_path, 'ImageParam.ini')
    config = configparser.ConfigParser(interpolation=None)
    config.read(ini_path, encoding='utf-8')

    # 2. 获取图像参数
    width = int(config['RawImageInfo']['Width'])
    height = int(config['RawImageInfo']['Height'])
    begin_index = int(config['FileModule']['BeginIndex'])
    end_index = int(config['FileModule']['EndIndex'])
    template = config['FileModule']['NameTemplate']

    # 像素数据类型
    bits_allocated = int(config['RawImageInfo']['BitsAllocated'])
    pixel_repr = int(config['RawImageInfo']['PixelRepresentation'])

    # 3. 推断 numpy dtype
    if bits_allocated == 16:
        if pixel_repr == 0:
            dtype = np.uint16
        else:
            dtype = np.int16
    elif bits_allocated == 8:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported BitsAllocated: {bits_allocated}")
    dtype = np.uint16
    # 4. 初始化体数据数组
    depth = end_index - begin_index + 1
    volume = np.zeros((depth, height, width), dtype=dtype)

    # 5. 读取每个切片
    for i in range(begin_index, end_index + 1):
        filename = template % i
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            slice_data = np.frombuffer(f.read(), dtype=dtype)
            if slice_data.size != width * height:
                raise ValueError(f"Slice {filename} size mismatch")
            volume[i - begin_index] = slice_data.reshape((height, width))

    return volume


def find_autofocus_slice(volume):
    focus_scores = []
    for i in range(volume.shape[0]):
        slice_img = volume[i]

        # 转换为 8bit 再计算 Laplacian（更稳定）
        if slice_img.dtype != np.uint8:
            norm_slice = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        else:
            norm_slice = slice_img

        laplacian = cv2.Laplacian(norm_slice, cv2.CV_64F)
        focus_measure = laplacian.var()
        focus_scores.append(focus_measure)

    best_index = int(np.argmax(focus_scores))
    return best_index, focus_scores

def find_peaks_in_nonzero_segments(data):
    peaks = []
    n = len(data)
    i = 0
    
    while i < n:
        # 跳过零值，寻找非零段的起始
        while i < n and data[i] == 0:
            i += 1
        if i >= n:
            break
        
        # 记录非零段的起始索引
        start = i
        # 寻找非零段的结束
        while i < n and data[i] != 0:
            i += 1
        end = i - 1
        
        # 在当前非零段中找峰值（最大值及其索引）
        segment = data[start:end+1]
        # print(segment)
        max_value = max(segment)
        max_index = np.argmax(segment) + start
        # max_index = segment.index(max_value) + start  # 转换为全局索引
        
        peaks.append(max_index)
    
    return peaks

def auto_focus(volume_data):
    best_idx, scores = find_autofocus_slice(volume_data)
    bin_list = []
    for i in range(len(scores)):
        # print(i)
        if scores[i] > 100:
            #  print( f"最清晰的焦点层为第 {i+1} 层,得分为:", scores[i])
            bin_list.append(scores[i])
        else:
            bin_list.append( 0)
    peak_idx_list = find_peaks_in_nonzero_segments(bin_list)
    return peak_idx_list
if __name__ == '__main__':
  # 示例用法
  root_path = r"D:\dataset\CT数据集\raw_data\Slice_t/"
  folder = root_path  # 修改为你的实际路径
  volume_data = load_volume_from_ini(folder)
  
  # 自动对焦
  peak_idx_list = auto_focus(volume_data)
  
  # 调用AI
  DEVICE = "cuda:0"
  WEIGHT_PATH = r"D:/Desktop/notebook/pipline/model/best.pt"
  model = ai_init( WEIGHT_PATH, DEVICE)

  model = model.eval()
  
  for idx in peak_idx_list:
    slice_data = volume_data[ idx, :, :]
    img = warpper_img(slice_data)
    result = inference(model, img, DEVICE)
    mask = get_mask( result, slice_data.shape)
    rect_show = show_result(result, img, slice_data.copy())
    plt.subplot(131)
    plt.imshow( slice_data, cmap='gray')
    plt.subplot(132)
    plt.imshow( rect_show, cmap='gray' )
    plt.subplot(133)
    plt.imshow( mask, cmap='gray' )
    plt.show()
