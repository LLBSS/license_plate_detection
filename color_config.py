import numpy as np

class PlateColorConfig:
    # 针对白色车牌的优化配置
    WHITE = {
        'name': '白色车牌',
        'hsv_ranges': [
            {
                'lower': np.array([0, 0, 120]),   # 适度亮度要求
                'upper': np.array([180, 50, 255]) # 适度的饱和度限制
            }
        ],
    }
    @classmethod
    def get_all_colors(cls):
        return {
            'white': cls.WHITE
        }
