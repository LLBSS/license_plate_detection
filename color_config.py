import numpy as np
import cv2

class PlateColorConfig:
    # 针对白色车牌的优化配置
    WHITE = {
        'name': '白色车牌',
        'hsv_ranges': [
            # 针对电动车白色车牌的优化范围
            {
                'lower': np.array([0, 0, 160]),   # 降低亮度阈值，捕捉更多白色区域
                'upper': np.array([180, 60, 255]) # 提高饱和度上限，避免捕捉过灰的区域
            },
            # 针对高光区域的补充范围
            {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            }
        ],
        'bgr_ranges': [
            # BGR空间的白色范围
            {
                'lower': np.array([150, 150, 150]),  # 降低下限
                'upper': np.array([255, 255, 255])
            }
        ]
    }
    
    # 蓝色车牌优化
    BLUE = {
        'name': '蓝色车牌',
        'hsv_ranges': [
            {
                'lower': np.array([90, 40, 40]),   # 放宽蓝色范围
                'upper': np.array([130, 255, 255])
            }
        ]
    }
    
    # 黄色车牌优化
    YELLOW = {
        'name': '黄色车牌',
        'hsv_ranges': [
            {
                'lower': np.array([10, 40, 40]),   # 放宽黄色范围
                'upper': np.array([40, 255, 255])
            }
        ]
    }
    
    # 绿色车牌优化
    GREEN = {
        'name': '绿色车牌',
        'hsv_ranges': [
            {
                'lower': np.array([30, 40, 40]),   # 放宽绿色范围
                'upper': np.array([90, 255, 255])
            }
        ]
    }
    
    @classmethod
    def get_all_colors(cls):
        return {
            'blue': cls.BLUE,
            'yellow': cls.YELLOW, 
            'green': cls.GREEN,
            'white': cls.WHITE
        }

class ColorRangeAdjuster:
    """颜色范围调整工具"""
    
    @staticmethod
    def adjust_white_range_based_on_image(image):
        """基于图像自适应调整白色范围"""
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 分析图像的亮度分布
        v_channel = hsv[:, :, 2]
        v_mean = np.mean(v_channel)
        v_std = np.std(v_channel)
        
        # 动态调整阈值
        if v_mean > 150:  # 高亮度图像
            lower_v = max(180, v_mean - v_std)
            upper_v = 255
            saturation_max = 50  # 降低饱和度要求
        else:  # 正常亮度图像
            lower_v = max(160, v_mean - v_std * 0.5)
            upper_v = 255
            saturation_max = 60
            
        return {
            'lower': np.array([0, 0, lower_v]),
            'upper': np.array([180, saturation_max, upper_v])
        }
    
    @staticmethod
    def create_dynamic_white_ranges(image):
        """创建动态的白色范围"""
        base_range = ColorRangeAdjuster.adjust_white_range_based_on_image(image)
        
        # 创建多个范围来覆盖不同光照条件
        ranges = [
            # 基础范围
            base_range,
            # 更宽松的范围
            {
                'lower': np.array([0, 0, base_range['lower'][2] - 20]),
                'upper': np.array([180, base_range['upper'][1] + 20, 255])
            },
            # 高亮度范围
            {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 40, 255])
            }
        ]
        
        return ranges

def get_optimized_color_config(image=None, target_color='white'):
    """获取优化后的颜色配置"""
    config = PlateColorConfig.get_all_colors()
    
    if target_color == 'white' and image is not None:
        # 为白色车牌创建动态范围
        dynamic_ranges = ColorRangeAdjuster.create_dynamic_white_ranges(image)
        config['white']['hsv_ranges'] = dynamic_ranges
    
    return config