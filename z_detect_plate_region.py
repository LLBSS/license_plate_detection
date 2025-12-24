# -*- coding: utf-8 -*-
import cv2
import numpy as np

def detect_plate_region(I, I_final):
    """
    车牌区域检测函数，对应MATLAB的detect_plate_region.m
    功能：根据预处理后的二值图像定位车牌区域，并进行二值化处理
    参数：I - 原始图像，I_final - 形态学滤波后的二值图像
    返回：I_plate - 二值化并去除边框后的车牌图像，I_plate_gray_out - 灰度车牌图像
    """
    I_plate = None
    I_plate_gray_out = None
    
    # 寻找二值图像中白色点的位置
    rows, cols = np.where(I_final == 255)  # OpenCV中白色是255
    location_of_1 = np.column_stack((rows, cols))
    
    if location_of_1.size > 0:
        # 找到车牌区域的最小外接矩形
        min_row = np.min(location_of_1[:, 0])
        max_row = np.max(location_of_1[:, 0])
        min_col = np.min(location_of_1[:, 1])
        max_col = np.max(location_of_1[:, 1])
        
        # 给边界留一些余量
        margin = 5
        x1 = max(0, min_row - margin)  # OpenCV的行索引从0开始
        x2 = min(I.shape[0], max_row + margin)
        y1 = max(0, min_col - margin)
        y2 = min(I.shape[1], max_col + margin)
        
        # 截取车牌区域
        I_plate_color = I[x1:x2, y1:y2, :]
        
        # 转换为灰度图
        if len(I_plate_color.shape) == 3:
            I_plate_gray = cv2.cvtColor(I_plate_color, cv2.COLOR_BGR2GRAY)
        else:
            I_plate_gray = I_plate_color
        
        # 保存灰度图用于输出显示
        I_plate_gray_out = I_plate_gray
        
        # 二值化 (OTSU) - 对于白色车牌黑色字符，需要反转二值化结果
        ret, I_plate_bw = cv2.threshold(I_plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 反转二值化结果：使黑色字符成为前景(255)，白色背景成为背景(0)
        # 注意：这里我们在threshold时已经使用了THRESH_BINARY_INV，所以不需要再反转
        
        # 去除小的噪点
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(I_plate_bw, connectivity=8)
        I_clean = np.zeros_like(I_plate_bw)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 50:
                I_clean[labels == i] = 255
        
        # 去除最大连通域（车牌外框）
        CC = cv2.connectedComponentsWithStats(I_clean, connectivity=8)
        num_objects = CC[0]
        stats = CC[2]
        
        if num_objects > 1:  # 至少有一个连通域
            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = np.argmax(areas) + 1  # 最大连通域的索引（0是背景）
            
            # 创建只包含最大连通域的掩码
            I_border = np.zeros_like(I_clean)
            I_border[CC[1] == idx] = 255
            
            # 从原图中减去边框
            I_plate_no_border = cv2.bitwise_and(I_clean, cv2.bitwise_not(I_border))
            I_plate = I_plate_no_border
        else:
            I_plate = I_clean
    
    return I_plate, I_plate_gray_out