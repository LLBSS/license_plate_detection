import cv2
import numpy as np

def adaptive_erode(binary_image):
    """
    自适应腐蚀函数 - 仅用于去除小噪声
    
    参数:
    binary_image: 输入的二值图像 (0和255)
    
    返回:
    kernel_size: 自适应计算的腐蚀核尺寸 (0-21之间的整数)
    """
    
    # 分析图像特征
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    if num_labels > 1:  # 确保有前景物体
        # 计算前景物体的平均宽度（跳过背景，索引0）
        avg_width = np.mean(stats[1:, cv2.CC_STAT_WIDTH])
        avg_height = np.mean(stats[1:, cv2.CC_STAT_HEIGHT])
    else:
        # 如果没有找到足够的前景组件，使用默认值
        avg_width, avg_height = 10, 10
    
    # 去除小噪声：核尺寸基于噪声尺度
    kernel_size = max(1, int(min(avg_width, avg_height) * 0.3))
    
    # 确保kernel_size在0到21之间
    kernel_size = max(0, min(21, kernel_size))
    
    return kernel_size

