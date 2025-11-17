import cv2
import numpy as np
from image_utils import preprocess_image, morphological_operations
from color_config import PlateColorConfig

class LicensePlateDetector:
    def __init__(self, use_color_detection=True, specific_color=None):
        """
        初始化车牌检测器
        
        Args:
            use_color_detection: 是否使用颜色检测
            specific_color: 指定车牌颜色，为None时检测所有颜色
        """
        # 放宽几何参数约束
        self.min_plate_width = 60
        self.max_plate_width = 400
        self.min_plate_height = 15
        self.max_plate_height = 150
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 4.5
        
        self.use_color_detection = use_color_detection
        self.specific_color = specific_color
        
        # 简化纹理特征参数
        self.min_edge_density = 0.03
        self.max_edge_density = 0.9
    
    def detect_plates_by_multiple_colors(self, image):
        """基于多种颜色检测车牌"""
        color_masks = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        colors_config = PlateColorConfig.get_all_colors()
        
        # 如果指定了特定颜色，只检测该颜色
        if self.specific_color and self.specific_color in colors_config:
            target_colors = {self.specific_color: colors_config[self.specific_color]}
        else:
            target_colors = colors_config
        
        for color_name, config in target_colors.items():
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # 使用HSV颜色空间检测
            for hsv_range in config['hsv_ranges']:
                mask = cv2.inRange(hsv, hsv_range['lower'], hsv_range['upper'])
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # 优化形态学操作参数
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # 先闭操作连接区域，再开操作去除噪点
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            
            color_masks[color_name] = combined_mask
        
        return color_masks
    
    def process_color_mask(self, color_mask):
        """专门处理颜色掩码，提取车牌区域"""
        # 复制掩码
        processed_mask = color_mask.copy()
        
        # 更强的形态学操作来连接车牌区域
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 闭操作连接车牌字符区域
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel_close)
        # 开操作去除小噪点
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel_open)
        
        return processed_mask
    
    def find_contours_in_mask(self, mask):
        """在掩码中查找轮廓"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def filter_and_validate_contours(self, contours, image):
        """过滤和验证轮廓"""
        valid_plates = []
        
        for contour in contours:
            # 使用边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算轮廓特征
            contour_area = cv2.contourArea(contour)
            rect_area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # 面积比例，衡量轮廓完整性
            area_ratio = contour_area / rect_area if rect_area > 0 else 0
            
            # 宽松的几何约束
            if (w >= self.min_plate_width and h >= self.min_plate_height and
                aspect_ratio >= self.min_aspect_ratio and aspect_ratio <= self.max_aspect_ratio and
                area_ratio >= 0.2):  # 降低完整性要求
                
                # 简化验证，主要信任颜色掩码
                if self.simple_region_validation(image, (x, y, w, h)):
                    valid_plates.append((x, y, w, h))
        
        return valid_plates
    
    def simple_region_validation(self, image, region):
        """简化的区域验证"""
        x, y, w, h = region
        
        # 基本尺寸检查
        if w < self.min_plate_width or h < self.min_plate_height:
            return False
        
        if w > self.max_plate_width or h > self.max_plate_height:
            return False
        
        # 宽高比检查
        aspect_ratio = w / h
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        # 简单边缘密度检查
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return self.min_edge_density <= edge_density <= self.max_edge_density
    
    def merge_similar_regions(self, regions, overlap_threshold=0.3):
        """合并相似区域"""
        if not regions:
            return []
        
        # 按面积排序
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        merged = []
        
        for current in regions:
            x1, y1, w1, h1 = current
            current_area = w1 * h1
            
            merged_flag = False
            
            for i, existing in enumerate(merged):
                x2, y2, w2, h2 = existing
                existing_area = w2 * h2
                
                # 计算重叠
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    
                    # 计算重叠比例
                    overlap_ratio = inter_area / min(current_area, existing_area)
                    
                    if overlap_ratio > overlap_threshold:
                        # 合并区域
                        new_x = min(x1, x2)
                        new_y = min(y1, y2)
                        new_w = max(x1 + w1, x2 + w2) - new_x
                        new_h = max(y1 + h1, y2 + h2) - new_y
                        
                        merged[i] = (new_x, new_y, new_w, new_h)
                        merged_flag = True
                        break
            
            if not merged_flag:
                merged.append(current)
        
        return merged
    
    def detect_plates_by_color(self, image, color_name, color_mask):
        """针对特定颜色进行车牌检测"""
        # 处理颜色掩码
        processed_mask = self.process_color_mask(color_mask)
        
        # 查找轮廓
        contours = self.find_contours_in_mask(processed_mask)
        
        # 过滤和验证轮廓
        plates = self.filter_and_validate_contours(contours, image)
        
        return plates
    
    def detect(self, image):
        """主检测函数"""
        all_plates = []
        
        if self.use_color_detection:
            # 获取所有颜色掩码
            color_masks = self.detect_plates_by_multiple_colors(image)
            
            # 对每种颜色进行检测
            for color_name, color_mask in color_masks.items():
                plates = self.detect_plates_by_color(image, color_name, color_mask)
                all_plates.extend(plates)
        
        # 合并重叠区域
        merged_plates = self.merge_similar_regions(all_plates)
        
        # 最终验证和排序
        final_plates = []
        for plate in merged_plates:
            if self.simple_region_validation(image, plate):
                final_plates.append(plate)
        
        # 按x坐标排序
        final_plates.sort(key=lambda p: p[0])
        
        return final_plates
    
    def draw_boxes(self, image, plates, color_info=None):
        """在图像上绘制车牌边界框"""
        result_image = image.copy()
        
        # 定义不同颜色的框
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255)   # 紫色
        ]
        
        for i, (x, y, w, h) in enumerate(plates):
            color = colors[i % len(colors)]
            
            # 绘制矩形框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # 添加标签
            label = f"Plate {i+1}"
            if color_info and i < len(color_info):
                label += f" ({color_info[i]})"
            
            # 绘制标签背景
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_image, (x, y-text_size[1]-10), 
                         (x+text_size[0], y), color, -1)
            
            # 绘制标签文字
            cv2.putText(result_image, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def debug_detection(self, image, display_masks=False):
        """调试检测过程"""
        if not self.use_color_detection or not display_masks:
            return None
        
        color_masks = self.detect_plates_by_multiple_colors(image)
        debug_images = {}
        
        for color_name, original_mask in color_masks.items():
            # 显示处理前后的掩码对比
            processed_mask = self.process_color_mask(original_mask)
            
            # 创建对比图像
            comparison = np.hstack([original_mask, processed_mask])
            debug_images[color_name] = comparison
            
            # 显示该颜色检测到的车牌
            plates = self.detect_plates_by_color(image, color_name, original_mask)
            if plates:
                print(f"{color_name}颜色检测到{len(plates)}个车牌")
        
        return debug_images