# -*- coding: utf-8 -*-
import cv2
import numpy as np

def preprocess_plate(I):
    """
    图像预处理函数，对应MATLAB的preprocess_plate.m
    功能：对输入图像进行灰度化、边缘检测和形态学滤波
    参数：I - 输入图像
    返回：I_final - 形态学滤波后的二值图像，I_gray - 灰度化并增强对比度后的图像
    """
    # 灰度化
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        I_gray = I
    
    # 增强对比度，使黑字符在白车牌上更明显
    # 使用CLAHE自适应直方图均衡化增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    I_gray = clahe.apply(I_gray)
    
    # 边缘检测 - 使用Sobel算子，降低阈值增强边缘检测
    I_edge = cv2.Canny(I_gray, 50, 150)  # Sobel边缘检测
    
    # 腐蚀 - 使用更细的结构元素保留字符细节
    se = np.array([[1], [1], [1]], dtype=np.uint8)
    I_erode = cv2.erode(I_edge, se)
    
    # 闭运算（填充）- 调整结构元素大小以适应白色车牌区域
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))  # 调整宽高比以适应车牌形状
    I_close = cv2.morphologyEx(I_erode, cv2.MORPH_CLOSE, se)
    
    # 开运算（去除噪点）- 调整面积阈值以保留车牌区域
    # 使用连通区域分析去除小面积噪点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(I_close, connectivity=8)
    I_final = np.zeros_like(I_close)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1500:
            I_final[labels == i] = 255
    
    return I_final, I_gray