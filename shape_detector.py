import cv2
import numpy as np
import math

class ShapeDetector:
    def __init__(self):
        self.min_contour_area = 500
        self.max_contour_area = 10000
        
    def detect_rectangles(self, edges, min_aspect_ratio=1.5, max_aspect_ratio=5.0):
        """检测矩形和平行四边形区域"""
        potential_regions = []
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
                
            # 方法1: 最小外接矩形（可检测倾斜矩形）
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算宽高比
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # 转换为边界矩形
                x, y, w, h = cv2.boundingRect(box)
                potential_regions.append(('rotated_rect', (x, y, w, h), aspect_ratio, rect))
            
            # 方法2: 边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # 计算矩形度（轮廓面积与边界矩形面积之比）
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0
                
                if rectangularity > 0.6:  # 较高的矩形度表明是规则形状
                    potential_regions.append(('bounding_rect', (x, y, w, h), aspect_ratio, None))
        
        return potential_regions
    
    def detect_parallelograms(self, edges, min_aspect_ratio=1.5, max_aspect_ratio=5.0):
        """检测平行四边形特征"""
        parallelograms = []
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 检测四边形
            if len(approx) == 4:
                # 检查是否接近平行四边形
                if self.is_parallelogram(approx):
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                        parallelograms.append(('parallelogram', (x, y, w, h), aspect_ratio, approx))
        
        return parallelograms
    
    def is_parallelogram(self, points):
        """判断四点是否构成平行四边形"""
        if len(points) != 4:
            return False
        
        # 将点排序为顺时针顺序
        points = self.sort_points_clockwise(points)
        
        # 计算向量
        vec1 = points[1] - points[0]
        vec2 = points[2] - points[1]
        vec3 = points[3] - points[2]
        vec4 = points[0] - points[3]
        
        # 检查对边是否平行
        parallel1 = self.is_parallel(vec1, vec3)
        parallel2 = self.is_parallel(vec2, vec4)
        
        return parallel1 and parallel2
    
    def is_parallel(self, vec1, vec2, angle_threshold=15):
        """判断两个向量是否平行"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return False
        
        # 计算夹角（角度）
        cos_angle = np.dot(vec1.flatten(), vec2.flatten()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.degrees(math.acos(cos_angle))
        
        # 允许的角度偏差
        return angle < angle_threshold or angle > 180 - angle_threshold
    
    def sort_points_clockwise(self, points):
        """将点按顺时针顺序排序"""
        points = points.reshape(4, 2)
        
        # 计算中心点
        center = np.mean(points, axis=0)
        
        # 计算每个点相对于中心点的角度
        angles = []
        for point in points:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            angles.append(math.atan2(dy, dx))
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]
    
    def combine_shape_features(self, image):
        """结合多种形状特征检测车牌"""
        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测各种形状
        rectangles = self.detect_rectangles(edges)
        parallelograms = self.detect_parallelograms(edges)
        
        # 合并结果
        all_shapes = rectangles + parallelograms
        
        # 去重和筛选
        filtered_shapes = self.filter_shape_regions(all_shapes)
        
        return filtered_shapes
    
    def filter_shape_regions(self, shapes, overlap_threshold=0.7):
        """过滤形状区域，去除重叠"""
        if not shapes:
            return []
        
        # 按面积排序
        shapes.sort(key=lambda s: s[1][2] * s[1][3], reverse=True)
        filtered = []
        
        for current in shapes:
            current_type, (x1, y1, w1, h1), aspect_ratio, extra = current
            current_area = w1 * h1
            
            is_duplicate = False
            
            for existing in filtered:
                existing_type, (x2, y2, w2, h2), _, _ = existing
                existing_area = w2 * h2
                
                # 计算重叠度
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    overlap_ratio = inter_area / min(current_area, existing_area)
                    
                    if overlap_ratio > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(current)
        
        return filtered