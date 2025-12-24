# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import signal

def segment_chars(I_plate):
    """
    字符分割函数，对应MATLAB的segment_chars.m
    功能：将车牌图像分割成单个字符
    参数：I_plate - 二值化并去除边框后的车牌图像
    返回：char_imgs - 分割后的字符图像列表，包含7个元素（标准车牌字符数）
    """
    # 初始化输出列表，预存7个空位置
    char_imgs = [None] * 7
    
    if I_plate is None:
        return char_imgs
    
    # 使用垂直投影法
    vertical_proj = np.sum(I_plate, axis=0)
    
    # 平滑投影曲线
    window_size = 3
    kernel = np.ones(window_size) / window_size
    vertical_proj_smooth = signal.convolve(vertical_proj, kernel, mode='same')
    
    # 动态阈值
    threshold = np.max(vertical_proj_smooth) * 0.1
    
    # 查找字符边界
    in_char = False
    char_start = []
    char_end = []
    
    for j in range(len(vertical_proj_smooth)):
        if vertical_proj_smooth[j] > threshold and not in_char:
            char_start.append(j)
            in_char = True
        elif vertical_proj_smooth[j] <= threshold and in_char:
            char_end.append(j)
            in_char = False
    
    if in_char:
        char_end.append(len(vertical_proj_smooth))
    
    # 过滤有效字符区域
    min_char_width = 5
    valid_chars = []
    
    for i in range(min(len(char_start), len(char_end))):
        width = char_end[i] - char_start[i]
        if width >= min_char_width:
            valid_chars.append(i)
    
    num_chars = min(7, len(valid_chars))
    
    if num_chars > 0:
        for n in range(num_chars):
            idx = valid_chars[n]
            
            # 提取字符
            left_bound = max(0, char_start[idx] - 1)
            right_bound = min(I_plate.shape[1], char_end[idx] + 1)
            
            char_img = I_plate[:, left_bound:right_bound]
            
            # 去除字符上下空白
            row_sum = np.sum(char_img, axis=1)
            non_zero_rows = np.where(row_sum > 0)[0]
            
            if non_zero_rows.size > 0:
                top = max(0, non_zero_rows[0] - 1)
                bottom = min(char_img.shape[0], non_zero_rows[-1] + 1)
                char_img = char_img[top:bottom, :]
            
            # 统一调整到标准尺寸 [32, 16]
            char_resized = cv2.resize(char_img, (16, 32), interpolation=cv2.INTER_NEAREST)
            char_imgs[n] = char_resized
        
        # 如果不足7个，填充空白
        for n in range(num_chars, 7):
            char_imgs[n] = np.zeros((32, 16), dtype=np.uint8)
    else:
        # 备用分割方法
        X = []
        flag = False
        for j in range(I_plate.shape[1]):
            sum_y = np.sum(I_plate[:, j])
            if (sum_y > 0) != flag:
                X.append(j)
                flag = (sum_y > 0)
        
        num_segments = min(7, len(X) // 2)
        for n in range(num_segments):
            if 2 * n + 1 < len(X):
                char_img = I_plate[:, X[2*n]:X[2*n+1]-1]
                
                # 去除上下空白
                row_sum = np.sum(char_img, axis=1)
                non_zero_rows = np.where(row_sum > 0)[0]
                if non_zero_rows.size > 0:
                    top = non_zero_rows[0]
                    bottom = non_zero_rows[-1]
                    char_img = char_img[top:bottom, :]
                
                # 调整到标准尺寸
                char_resized = cv2.resize(char_img, (16, 32), interpolation=cv2.INTER_NEAREST)
                char_imgs[n] = char_resized
        
        # 填充剩余空位
        for n in range(num_segments, 7):
            char_imgs[n] = np.zeros((32, 16), dtype=np.uint8)
    
    return char_imgs