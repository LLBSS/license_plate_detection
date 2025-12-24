import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class LicensePlateDetector:
    """
    检测彩色图像中位于中间位置的白色矩形框（车牌）
    特点：
    1. 检测白色或近白色矩形
    2. 主要关注图像中间区域
    3. 支持透视变换矫正
    """
    
    def __init__(self, debug: bool = False):
        """
        初始化检测器
        
        参数:
            debug: 是否显示调试信息
        """
        self.debug = debug
        
        # 车牌常见颜色范围 (HSV空间)
        self.color_ranges = {
            'white': ([0, 0, 200], [180, 30, 255]),      # 白色/银色车牌
            'blue': ([100, 150, 50], [130, 255, 255]),   # 蓝色车牌
            'green': ([40, 100, 50], [80, 255, 255]),    # 绿色新能源车牌
            'yellow': ([20, 100, 100], [40, 255, 255]),  # 黄色车牌
        }
        
        # 车牌尺寸比例约束
        self.plate_aspect_ratio_range = (2.0, 5.0)  # 长宽比范围
        self.min_plate_area_ratio = 0.005           # 最小面积占图像面积比例
        self.max_plate_area_ratio = 0.2             # 最大面积占图像面积比例
        
    def detect_white_rectangle(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测图像中间的白色矩形框（车牌）
        
        参数:
            image: 输入彩色图像 (BGR格式)
            
        返回:
            dict: 包含检测结果的字典，或None（如果未检测到）
        """
        if image is None:
            raise ValueError("输入图像为空")
            
        original = image.copy()
        height, width = image.shape[:2]
        
        # 1. 预处理图像
        processed = self._preprocess_image(image)
        
        # 2. 检测白色区域
        white_mask = self._detect_white_regions(image)
        
        if self.debug:
            cv2.imshow("White Mask", white_mask)
            cv2.waitKey(0)
        
        # 3. 查找轮廓
        contours = self._find_contours(white_mask)
        
        if not contours:
            print("未找到轮廓")
            return None
        
        # 4. 筛选可能的车牌轮廓
        plates = self._filter_plate_contours(contours, width, height)
        
        if not plates:
            print("未找到符合条件的车牌轮廓")
            return None
        
        # 5. 选择最佳车牌区域
        best_plate = self._select_best_plate(plates, width, height)
        
        if best_plate is None:
            return None
        
        # 6. 获取最小外接矩形
        plate_rect = cv2.minAreaRect(best_plate['contour'])
        plate_box = cv2.boxPoints(plate_rect)
        plate_box = np.intp(plate_box)
        
        # 7. 计算矩形属性
        center_x, center_y = self._calculate_contour_center(best_plate['contour'])
        distance_from_center = np.sqrt((center_x - width/2)**2 + (center_y - height/2)**2)
        
        result = {
            'box_points': plate_box,            # 矩形四个顶点
            'min_rect': plate_rect,             # 最小外接矩形参数
            'contour': best_plate['contour'],   # 轮廓点
            'center': (center_x, center_y),     # 矩形中心
            'area': best_plate['area'],         # 面积
            'aspect_ratio': best_plate['aspect_ratio'],  # 长宽比
            'distance_from_center': distance_from_center,  # 距离图像中心的距离
            'confidence': self._calculate_confidence(best_plate, width, height)
        }
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        """
        # 转换为HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        return blurred
    
    def _detect_white_regions(self, image: np.ndarray) -> np.ndarray:
        """
        检测白色区域
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 白色范围1: 低饱和度，高亮度
        lower_white1 = np.array([0, 0, 200])
        upper_white1 = np.array([180, 30, 255])
        mask1 = cv2.inRange(hsv, lower_white1, upper_white1)
        
        # 白色范围2: 扩展范围
        lower_white2 = np.array([0, 0, 180])
        upper_white2 = np.array([180, 60, 255])
        mask2 = cv2.inRange(hsv, lower_white2, upper_white2)
        
        # 合并掩码
        white_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # 应用中间区域掩码（只关注中间区域）
        center_mask = self._create_center_mask(image.shape)
        white_mask = cv2.bitwise_and(white_mask, center_mask)
        
        return white_mask
    
    def _create_center_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建中间区域掩码
        """
        height, width = image_shape[:2]
        
        # 创建全黑掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 定义中间区域（图像中心的1/3区域）
        center_x, center_y = width // 2, height // 2
        center_width, center_height = width // 3, height // 3
        
        # 绘制白色矩形
        x1 = center_x - center_width // 2
        y1 = center_y - center_height // 2
        x2 = center_x + center_width // 2
        y2 = center_y + center_height // 2
        
        # 确保坐标不超出图像边界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        查找轮廓
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _filter_plate_contours(self, contours: List[np.ndarray], 
                              image_width: int, image_height: int) -> List[Dict]:
        """
        筛选可能的车牌轮廓
        """
        plates = []
        min_area = image_width * image_height * self.min_plate_area_ratio
        max_area = image_width * image_height * self.max_plate_area_ratio
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < min_area or area > max_area:
                continue
            
            # 计算轮廓的外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # 计算矩形尺寸
            width = rect[1][0]
            height = rect[1][1]
            
            # 确保宽度大于高度
            if height > width:
                width, height = height, width
            
            # 计算长宽比
            aspect_ratio = width / (height + 1e-6)
            
            # 长宽比过滤
            if not (self.plate_aspect_ratio_range[0] <= aspect_ratio <= self.plate_aspect_ratio_range[1]):
                continue
            
            # 计算轮廓中心
            center_x, center_y = self._calculate_contour_center(contour)
            
            # 确保在图像中间区域
            if not self._is_in_center_region(center_x, center_y, image_width, image_height):
                continue
            
            # 计算轮廓的矩形度
            rect_area = width * height
            extent = area / rect_area if rect_area > 0 else 0
            
            # 矩形度过滤（接近1表示更接近矩形）
            if extent < 0.6:
                continue
            
            plates.append({
                'contour': contour,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center': (center_x, center_y),
                'rect': rect,
                'box': box,
                'extent': extent
            })
        
        return plates
    
    def _calculate_contour_center(self, contour: np.ndarray) -> Tuple[float, float]:
        """
        计算轮廓中心
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            # 如果矩为0，使用外接矩形中心
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
        return center_x, center_y
    
    def _is_in_center_region(self, x: int, y: int, 
                            image_width: int, image_height: int) -> bool:
        """
        判断点是否在图像中间区域
        """
        center_region_width = image_width // 2
        center_region_height = image_height // 2
        
        center_x = image_width // 2
        center_y = image_height // 2
        
        left = center_x - center_region_width // 2
        right = center_x + center_region_width // 2
        top = center_y - center_region_height // 2
        bottom = center_y + center_region_height // 2
        
        return left <= x <= right and top <= y <= bottom
    
    def _select_best_plate(self, plates: List[Dict], 
                          image_width: int, image_height: int) -> Optional[Dict]:
        """
        选择最佳的车牌区域
        """
        if not plates:
            return None
        
        # 计算每个车牌的得分
        for plate in plates:
            # 基础得分
            score = 0
            
            # 面积得分（适中最好）
            area_ratio = plate['area'] / (image_width * image_height)
            if 0.01 <= area_ratio <= 0.1:  # 面积占比1%-10%
                score += 2
            elif 0.005 <= area_ratio <= 0.15:  # 面积占比0.5%-15%
                score += 1
            
            # 长宽比得分（接近3.14最佳）
            aspect_ratio = plate['aspect_ratio']
            if 2.8 <= aspect_ratio <= 3.3:  # 接近标准车牌长宽比3.14
                score += 3
            elif 2.5 <= aspect_ratio <= 3.5:
                score += 2
            elif 2.0 <= aspect_ratio <= 4.0:
                score += 1
            
            # 中心距离得分（越中心越高）
            center_x, center_y = plate['center']
            distance = np.sqrt((center_x - image_width/2)**2 + (center_y - image_height/2)**2)
            max_distance = np.sqrt((image_width/2)**2 + (image_height/2)**2)
            distance_score = 1 - (distance / max_distance)
            score += distance_score * 2
            
            # 矩形度得分
            score += plate['extent']
            
            plate['score'] = score
        
        # 选择得分最高的
        best_plate = max(plates, key=lambda x: x['score'])
        
        return best_plate if best_plate['score'] > 0 else None
    
    def _calculate_confidence(self, plate: Dict, 
                            image_width: int, image_height: int) -> float:
        """
        计算检测置信度
        """
        confidence = 0.0
        
        # 面积占比得分
        area_ratio = plate['area'] / (image_width * image_height)
        if 0.01 <= area_ratio <= 0.1:
            confidence += 0.3
        
        # 长宽比得分
        aspect_ratio = plate['aspect_ratio']
        if 2.8 <= aspect_ratio <= 3.3:
            confidence += 0.3
        elif 2.5 <= aspect_ratio <= 3.5:
            confidence += 0.2
        
        # 中心距离得分
        center_x, center_y = plate['center']
        distance = np.sqrt((center_x - image_width/2)**2 + (center_y - image_height/2)**2)
        max_distance = np.sqrt((image_width/2)**2 + (image_height/2)**2)
        if distance < max_distance * 0.3:
            confidence += 0.2
        
        # 矩形度得分
        confidence += plate['extent'] * 0.2
        
        return min(confidence, 1.0)
    
    def draw_detection(self, image: np.ndarray, 
                      detection_result: Dict) -> np.ndarray:
        """
        在图像上绘制检测结果
        """
        result_image = image.copy()
        
        # 绘制检测框
        box_points = detection_result['box_points']
        cv2.drawContours(result_image, [box_points], 0, (0, 255, 0), 3)
        
        # 绘制中心点
        center_x, center_y = detection_result['center']
        cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 绘制图像中心
        height, width = image.shape[:2]
        cv2.circle(result_image, (width//2, height//2), 5, (255, 0, 0), -1)
        cv2.line(result_image, (width//2, height//2), (center_x, center_y), (255, 0, 0), 2)
        
        # 添加文本信息
        info = [
            f"Confidence: {detection_result['confidence']:.2f}",
            f"Aspect Ratio: {detection_result['aspect_ratio']:.2f}",
            f"Area: {detection_result['area']}",
            f"Center Distance: {detection_result['distance_from_center']:.1f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info):
            cv2.putText(result_image, text, (10, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_image
    
    def extract_plate(self, image: np.ndarray, 
                     detection_result: Dict) -> np.ndarray:
        """
        提取车牌区域
        """
        # 获取矩形四个点
        src_points = detection_result['box_points'].astype(np.float32)
        
        # 对点进行排序（左上、右上、右下、左下）
        src_points = self._order_points(src_points)
        
        # 计算变换后的宽度和高度
        width = int(max(
            np.linalg.norm(src_points[0] - src_points[1]),
            np.linalg.norm(src_points[2] - src_points[3])
        ))
        height = int(max(
            np.linalg.norm(src_points[0] - src_points[3]),
            np.linalg.norm(src_points[1] - src_points[2])
        ))
        
        # 目标点
        dst_points = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype=np.float32)
        
        # 透视变换
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        对四个点进行排序：左上、右上、右下、左下
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 左上点：x+y最小
        # 右下点：x+y最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 右上点：y-x最小
        # 左下点：y-x最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect


# 简化易用版函数
def detect_white_license_plate(image_path: str, 
                              show_result: bool = True,
                              save_result: bool = False) -> Optional[Dict]:
    """
    简化版函数：检测白色车牌
    
    参数:
        image_path: 图像路径
        show_result: 是否显示结果
        save_result: 是否保存结果
        
    返回:
        检测结果字典或None
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 创建检测器
    detector = LicensePlateDetector(debug=False)
    
    # 检测车牌
    result = detector.detect_white_rectangle(image)
    
    if result is None:
        print("未检测到车牌")
        return None
    
    # 提取车牌区域
    plate_image = detector.extract_plate(image, result)
    
    # 绘制检测结果
    result_image = detector.draw_detection(image, result)
    
    # 显示结果
    if show_result:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f"检测结果 (置信度: {result['confidence']:.2f})")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
        plt.title("提取的车牌")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 保存结果
    if save_result:
        cv2.imwrite("detection_result.jpg", result_image)
        cv2.imwrite("extracted_plate.jpg", plate_image)
        print("结果已保存")
    
    # 打印结果信息
    print("=" * 50)
    print("车牌检测结果:")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"  中心坐标: {result['center']}")
    print(f"  长宽比: {result['aspect_ratio']:.2f}")
    print(f"  面积: {result['area']} 像素")
    print(f"  距离图像中心: {result['distance_from_center']:.1f} 像素")
    print("=" * 50)
    
    return result


# 批量处理函数
def detect_plates_in_folder(folder_path: str, 
                          output_folder: str = "output") -> List[Dict]:
    """
    批量处理文件夹中的图像
    
    参数:
        folder_path: 输入文件夹路径
        output_folder: 输出文件夹路径
        
    返回:
        所有检测结果列表
    """
    import os
    from glob import glob
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
    
    print(f"找到 {len(image_files)} 张图像")
    
    all_results = []
    detector = LicensePlateDetector(debug=False)
    
    for image_file in image_files:
        print(f"\n处理: {os.path.basename(image_file)}")
        
        # 读取图像
        image = cv2.imread(image_file)
        if image is None:
            print(f"  无法读取: {image_file}")
            continue
        
        # 检测车牌
        result = detector.detect_white_rectangle(image)
        
        if result is None:
            print(f"  未检测到车牌")
            continue
        
        # 提取和保存结果
        filename = os.path.splitext(os.path.basename(image_file))[0]
        
        # 绘制检测框
        result_image = detector.draw_detection(image, result)
        cv2.imwrite(os.path.join(output_folder, f"{filename}_detected.jpg"), result_image)
        
        # 提取车牌
        plate_image = detector.extract_plate(image, result)
        cv2.imwrite(os.path.join(output_folder, f"{filename}_plate.jpg"), plate_image)
        
        print(f"  检测成功! 置信度: {result['confidence']:.2%}")
        
        all_results.append({
            'file': image_file,
            'result': result,
            'plate_image': plate_image
        })
    
    print(f"\n处理完成! 成功检测 {len(all_results)} 个车牌")
    
    return all_results


# 创建测试图像的实用函数
def create_test_image_with_white_plate():
    """
    创建包含白色车牌的测试图像
    """
    # 创建随机背景
    img = np.random.randint(50, 150, (400, 600, 3), dtype=np.uint8)
    
    # 在中间位置添加白色矩形（模拟车牌）
    plate_width, plate_height = 200, 60
    center_x, center_y = 300, 200
    
    # 随机倾斜角度
    angle = np.random.uniform(-15, 15)
    
    # 创建旋转矩阵
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # 创建白色车牌
    plate = np.ones((plate_height, plate_width, 3), dtype=np.uint8) * 255
    
    # 添加车牌文字
    cv2.putText(plate, '京A·12345', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 旋转车牌
    plate_rotated = cv2.warpAffine(plate, M, (600, 400), 
                                  borderMode=cv2.BORDER_TRANSPARENT)
    
    # 将车牌叠加到背景
    mask = plate_rotated > 200
    img[mask] = plate_rotated[mask]
    
    # 添加一些噪声
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # 保存测试图像
    cv2.imwrite("test_white_plate.jpg", img)
    print("测试图像已创建: test_white_plate.jpg")
    
    return img

def cv2_to_pil_image(cv2_image):
    """
    将OpenCV图像转换为PIL图像
    
    参数:
    cv2_image: OpenCV图像(BGR格式)
    
    返回:
    PIL图像(RGB格式)
    """
    # 转换颜色空间
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def resize_image(image, max_width=800, max_height=600):
    """
    调整图像大小以适应显示窗口
    
    参数:
    image: OpenCV图像
    max_width: 最大宽度
    max_height: 最大高度
    
    返回:
    调整大小后的图像
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale >= 1:
        return image
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def draw_fit_lines(image_path):
    """
    绘制车牌边界线
    
    参数:
    image_path: 图像路径
    
    返回:
    绘制了边界线的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    # 保存原始图像
    original = img.copy()
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"未找到轮廓: {image_path}")
        return None
    
    # 按面积排序，获取最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    
    # 获取轮廓的最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # 提取轮廓的所有点
    points = largest_contour.reshape(-1, 2)
    
    # 根据y坐标将点分为上下两部分
    center_y = np.mean(points[:, 1])
    upper_points = points[points[:, 1] < center_y]
    lower_points = points[points[:, 1] >= center_y]
    
    if len(upper_points) < 2 or len(lower_points) < 2:
        print(f"轮廓点不足，无法拟合线条: {image_path}")
        return None
    
    # 使用cv2.fitLine拟合上下边界线
    [vx1, vy1, x1, y1] = cv2.fitLine(upper_points, cv2.DIST_L2, 0, 0.01, 0.01)
    [vx2, vy2, x2, y2] = cv2.fitLine(lower_points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 计算线条的起点和终点
    height, width = img.shape[:2]
    
    # 上边界线
    lefty1 = int((-x1 * vy1 / vx1) + y1)
    righty1 = int(((width - x1) * vy1 / vx1) + y1)
    
    # 下边界线
    lefty2 = int((-x2 * vy2 / vx2) + y2)
    righty2 = int(((width - x2) * vy2 / vx2) + y2)
    
    # 绘制边界线
    cv2.line(img, (width - 1, righty1), (0, lefty1), (0, 255, 0), 2)
    cv2.line(img, (width - 1, righty2), (0, lefty2), (0, 255, 0), 2)
    
    return img


class LicensePlateApp:
    """
    车牌边界线拟合可视化应用
    """
    def __init__(self, root):
        self.root = root
        self.root.title("车牌边界线拟合")
        self.root.geometry("1000x700")
        
        # 设置test文件夹路径
        self.test_folder = "test"
        self.image_files = []
        self.current_image = None
        self.current_result = None
        
        # 创建界面布局
        self.create_widgets()
        
        # 加载图片列表
        self.load_image_list()
    
    def create_widgets(self):
        """
        创建界面组件
        """
        # 创建顶部框架
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 创建标题
        title_label = ttk.Label(top_frame, text="车牌边界线拟合", font=("Arial", 16))
        title_label.pack()
        
        # 创建中间框架
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧图片选择区域
        left_frame = ttk.LabelFrame(middle_frame, text="选择图片")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 创建图片列表
        self.image_listbox = tk.Listbox(left_frame, width=30, height=20)
        self.image_listbox.pack(fill=tk.Y, expand=True, padx=5, pady=5)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        
        # 绑定选择事件
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)
        
        # 创建右侧图片展示区域
        right_frame = ttk.LabelFrame(middle_frame, text="图片预览")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建画布
        self.canvas = tk.Canvas(right_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建底部框架
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 创建保存按钮
        self.save_button = ttk.Button(bottom_frame, text="保存结果", command=self.save_result, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # 创建状态标签
        self.status_label = ttk.Label(bottom_frame, text="请选择一张图片")
        self.status_label.pack(side=tk.LEFT, padx=5)
    
    def load_image_list(self):
        """
        加载test文件夹中的图片列表
        """
        # 清空列表
        self.image_listbox.delete(0, tk.END)
        
        # 检查test文件夹是否存在
        if not os.path.exists(self.test_folder):
            self.status_label.config(text="错误: test文件夹不存在")
            return
        
        # 获取test文件夹中的所有图片文件
        self.image_files = [f for f in os.listdir(self.test_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not self.image_files:
            self.status_label.config(text="错误: test文件夹中没有图片文件")
            return
        
        # 添加到列表框
        for file in self.image_files:
            self.image_listbox.insert(tk.END, file)
        
        self.status_label.config(text=f"已加载 {len(self.image_files)} 张图片")
    
    def on_image_select(self, event):
        """
        当选择图片时的处理函数
        """
        # 获取选中的索引
        selection = self.image_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        selected_file = self.image_files[index]
        self.status_label.config(text=f"正在处理: {selected_file}")
        
        # 获取图片路径
        image_path = os.path.join(self.test_folder, selected_file)
        
        # 绘制边界线
        result = draw_fit_lines(image_path)
        
        if result is not None:
            self.current_image = selected_file
            self.current_result = result
            
            # 调整展示大小
            resized_result = resize_image(result)
            
            # 显示结果
            self.display_image(resized_result)
            
            # 启用保存按钮
            self.save_button.config(state=tk.NORMAL)
            
            self.status_label.config(text=f"处理完成: {selected_file}")
        else:
            self.status_label.config(text=f"处理失败: {selected_file}")
    
    def display_image(self, cv2_image):
        """
        在画布上显示图像
        
        参数:
        cv2_image: OpenCV图像
        """
        # 转换为PIL图像
        pil_image = cv2_to_pil_image(cv2_image)
        
        # 转换为PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 清空画布
        self.canvas.delete("all")
        
        # 获取画布大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 获取图像大小
        image_width = self.photo.width()
        image_height = self.photo.height()
        
        # 计算图像位置（居中显示）
        x = (canvas_width - image_width) // 2
        y = (canvas_height - image_height) // 2
        
        # 显示图像
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
    
    def save_result(self):
        """
        保存处理结果
        """
        if self.current_result is None or self.current_image is None:
            return
        
        # 保存结果到test文件夹
        output_name = f"{os.path.splitext(self.current_image)[0]}_with_lines{os.path.splitext(self.current_image)[1]}"
        output_path = os.path.join(self.test_folder, output_name)
        
        if cv2.imwrite(output_path, self.current_result):
            self.status_label.config(text=f"结果已保存到: {output_name}")
        else:
            self.status_label.config(text="保存失败")


if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    
    # 创建应用实例
    app = LicensePlateApp(root)
    
    # 启动主循环
    root.mainloop()
    # results = detect_plates_in_folder("input_images/", "output/")