import cv2
import numpy as np
import matplotlib.pyplot as plt

def refine_canny_edges(canny_img, min_contour_length=100, kernel_size=3, 
                       area_threshold=500, close_iterations=2):
    # 确保输入是二值图像
    if len(canny_img.shape) == 3:
        canny_img = cv2.cvtColor(canny_img, cv2.COLOR_BGR2GRAY)
    # 1. 形态学操作 - 连接断开的边缘
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 先闭运算连接断开的边缘
    closed = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    # 2. 查找轮廓
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 3. 创建空白画布
    refined_edges = np.zeros_like(canny_img)
    # 4. 筛选轮廓
    for contour in contours:
        # 计算轮廓长度
        contour_length = cv2.arcLength(contour, closed=True)
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        # 筛选条件：轮廓足够长且面积足够大
        if contour_length >= min_contour_length and area >= area_threshold:
            # 可选：计算凸包来平滑轮廓
            hull = cv2.convexHull(contour)
            # 绘制筛选后的轮廓
            cv2.drawContours(refined_edges, [hull], 0, 255, 1)
            #cv2.drawContours(refined_edges, [contour], 0, 255, 1)
    # 5. 可选：对结果进行细化处理
    # 使用非极大值抑制细化边缘
    refined_edges = cv2.ximgproc.thinning(refined_edges)
    return refined_edges
# 使用示例
if __name__ == "__main__":
    canny_img = cv2.imread('examples-canny\\canny_car (1).jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Canny Image', canny_img)
    refined = refine_canny_edges(canny_img)
    cv2.imwrite('car1-canny.jpg', refined)