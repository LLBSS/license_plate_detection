import cv2
import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
import matplotlib

# 解决Windows中文乱码问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def process_license_plate(image_path):
    """
    处理车牌图片：矫正倾斜 + 提取车牌区域
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片")
    
    original = img.copy()
    height, width = img.shape[:2]
    
    # 2. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 3. 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=50, maxLineGap=10)
    
    # 4. 分析直线角度
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            if abs(angle) < 80 or abs(angle) > 100:
                angles.append(angle)
    
    # 5. 计算旋转角度
    if len(angles) > 0:
        angles = np.array(angles)
        rotation_angle = np.median(angles)
        
        if rotation_angle > 45:
            rotation_angle -= 90
        elif rotation_angle < -45:
            rotation_angle += 90
    else:
        rotation_angle = 0
    
    # 6. 旋转矫正
    if abs(rotation_angle) > 0.5:
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        corrected_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(0, 0, 0))
    else:
        corrected_img = img.copy()
    
    # 7. 检测并精确提取车牌区域
    plate_only = extract_plate_region(corrected_img)
    
    return original, corrected_img, plate_only, rotation_angle

def extract_plate_region(corrected_img):
    """
    精确提取车牌区域，去除背景
    """
    # 转换为HSV颜色空间，更好地分离颜色
    hsv = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2HSV)
    
    # 定义蓝色车牌的HSV范围（中国车牌常见颜色）
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # 定义黄色车牌的HSV范围
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # 定义绿色车牌的HSV范围
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # 定义白色车牌的HSV范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    
    # 尝试检测不同颜色的车牌
    masks = []
    for lower, upper in [(lower_blue, upper_blue), (lower_yellow, upper_yellow), (lower_green, upper_green), (lower_white, upper_white)]:
        mask = cv2.inRange(hsv, lower, upper)
        masks.append(mask)
    
    # 合并所有颜色掩码
    color_mask = cv2.bitwise_or(masks[0], masks[1])
    color_mask = cv2.bitwise_or(color_mask, masks[2])
    color_mask = cv2.bitwise_or(color_mask, masks[3])
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 使用边缘检测作为备用方案
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, iterations=2)
    
    # 合并颜色和边缘信息
    combined_mask = cv2.bitwise_or(color_mask, edges)
    
    # 查找轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 寻找最佳车牌轮廓
    best_contour = None
    best_score = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算特征
        aspect_ratio = w / h if h > 0 else 0
        area = w * h
        solidity = area / (cv2.contourArea(contour) + 1e-6)
        
        # 车牌特征评分
        score = 0
        
        # 宽高比评分（车牌通常为3.14:1左右）
        if 2.5 < aspect_ratio < 4.5:
            score += 3
        elif 2.0 < aspect_ratio < 5.0:
            score += 1
        
        # 面积评分
        img_area = corrected_img.shape[0] * corrected_img.shape[1]
        if 0.01 < area / img_area < 0.3:  # 车牌面积占图像的1%-30%
            score += 2
        
        # 实心度评分
        if solidity > 0.6:
            score += 1
        
        # 更新最佳轮廓
        if score > best_score:
            best_score = score
            best_contour = contour
    
    # 如果没有找到合适的轮廓，尝试使用灰度阈值
    if best_contour is None or best_score < 3:
        # 使用自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if 2.5 < aspect_ratio < 4.5 and area > 1000:
                score = 3
                if score > best_score:
                    best_score = score
                    best_contour = contour
    
    # 提取车牌区域
    if best_contour is not None and best_score >= 3:
        # 获取最小外接矩形
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # 排序四个点
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect
        
        ordered_box = order_points(box)
        
        # 计算宽度和高度
        widthA = np.linalg.norm(ordered_box[0] - ordered_box[1])
        widthB = np.linalg.norm(ordered_box[2] - ordered_box[3])
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(ordered_box[0] - ordered_box[3])
        heightB = np.linalg.norm(ordered_box[1] - ordered_box[2])
        maxHeight = max(int(heightA), int(heightB))
        
        # 目标点
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # 透视变换
        M = cv2.getPerspectiveTransform(ordered_box, dst)
        plate_only = cv2.warpPerspective(corrected_img, M, (maxWidth, maxHeight))
        
        # 确保车牌方向正确（宽度>高度）
        if plate_only.shape[0] > plate_only.shape[1]:
            plate_only = cv2.rotate(plate_only, cv2.ROTATE_90_CLOCKWISE)
        
        # 添加边界
        border_size = 5
        plate_only = cv2.copyMakeBorder(plate_only, border_size, border_size, 
                                       border_size, border_size, 
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        return plate_only
    else:
        # 如果没有找到车牌，返回整个图片
        print("未检测到车牌区域，返回完整图片")
        return corrected_img

def visualize_results(original, corrected, plate_only, angle):
    """可视化处理结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图片
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'原始图片\n尺寸: {original.shape[1]}x{original.shape[0]}')
    axes[0].axis('off')
    
    # 矫正后的图片
    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'矫正后图片\n旋转角度: {angle:.2f}度')
    axes[1].axis('off')
    
    # 只包含车牌的图片
    axes[2].imshow(cv2.cvtColor(plate_only, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'提取的车牌\n尺寸: {plate_only.shape[1]}x{plate_only.shape[0]}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """主程序"""
    # 设置默认打开路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'test')
    
    if not os.path.exists(test_dir):
        test_dir = current_dir
        print(f"注意: 'test' 文件夹不存在，使用当前目录: {test_dir}")
    
    # 创建Tkinter根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 打开文件选择对话框
    image_path = filedialog.askopenfilename(
        title="选择车牌图片",
        initialdir=test_dir,
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
    )
    
    if not image_path:
        print("未选择图片，程序退出")
        return
    
    try:
        # 处理图片
        print("正在处理图片...")
        original, corrected, plate_only, angle = process_license_plate(image_path)
        
        # 显示结果
        visualize_results(original, corrected, plate_only, angle)
        
        # 询问是否保存结果
        root.deiconify()  # 显示窗口
        save_result = messagebox.askyesno("保存结果", "是否保存处理结果？")
        
        if save_result:
            # 获取保存路径
            save_dir = filedialog.askdirectory(
                title="选择保存文件夹",
                initialdir=os.path.dirname(image_path)
            )
            
            if save_dir:
                # 生成文件名
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                corrected_path = os.path.join(save_dir, f"{base_name}_corrected.jpg")
                # 保存图片
                cv2.imwrite(corrected_path, corrected)
                
                print(f"结果已保存:")
        root.destroy()
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        root.destroy()

if __name__ == "__main__":
    main()