import cv2
import numpy as np
from color_config import PlateColorConfig
class WhitePlateDetector:
     def detect_plates_by_white_colors(self, image):
        """基于白色颜色检测车牌"""
        color_masks = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        colors_config = PlateColorConfig.get_all_colors()
        
        for color_name, config in colors_config.items():
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # 使用HSV颜色空间检测
            for hsv_range in config['hsv_ranges']:
                mask = cv2.inRange(hsv, hsv_range['lower'], hsv_range['upper'])
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            color_masks[color_name] = combined_mask
        return color_masks