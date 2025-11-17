import cv2
import numpy as np

def load_image(image_path):
    """加载图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return image

def preprocess_image(image):
    """图像预处理"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    return gray, edges

def morphological_operations(image):
    """形态学操作增强车牌区域"""
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # 闭操作连接车牌区域
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # 开操作去除小噪点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened

def display_image(image, title="Image"):
    """显示图像（用于调试）"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()