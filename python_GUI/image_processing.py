import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class ImageProcessor:
    """图片处理类，负责应用各种滤镜和图像调整效果"""
    
    @staticmethod
    def apply_filter(image, filter_type):
        """应用各种滤镜效果"""
        if filter_type == "黑白":
            return image.convert('L')
        elif filter_type == "复古":
            # 复古滤镜 - 降低饱和度并偏暖色调
            img_array = np.array(image)
            # 降低饱和度
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # 增加暖色调（增加红色和黄色通道）
                img_array = img_array.copy()
                img_array[..., 0] = np.clip(img_array[..., 0] * 1.2, 0, 255)  # 红色通道
                img_array[..., 1] = np.clip(img_array[..., 1] * 1.1, 0, 255)  # 绿色通道
                return Image.fromarray(img_array.astype(np.uint8))
            return image
        elif filter_type == "冷色调":
            # 冷色调 - 增加蓝色通道
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                img_array = img_array.copy()
                img_array[..., 2] = np.clip(img_array[..., 2] * 1.2, 0, 255)  # 蓝色通道
                img_array[..., 0] = np.clip(img_array[..., 0] * 0.9, 0, 255)  # 减少红色通道
                return Image.fromarray(img_array.astype(np.uint8))
            return image
        elif filter_type == "暖色调":
            # 暖色调 - 增加红色和绿色通道
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                img_array = img_array.copy()
                img_array[..., 0] = np.clip(img_array[..., 0] * 1.2, 0, 255)  # 红色通道
                img_array[..., 1] = np.clip(img_array[..., 1] * 1.1, 0, 255)  # 绿色通道
                img_array[..., 2] = np.clip(img_array[..., 2] * 0.9, 0, 255)  # 减少蓝色通道
                return Image.fromarray(img_array.astype(np.uint8))
            return image
        elif filter_type == "锐化":
            return image.filter(ImageFilter.SHARPEN)
        elif filter_type == "模糊":
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        return image
    
    @staticmethod
    def process_image(original_image, params):
        """处理图片（应用各种效果）"""
        if original_image is None:
            return None
        
        # 获取当前图片参数
        brightness = params.get("brightness", 1.0)
        contrast = params.get("contrast", 1.0)
        saturation = params.get("saturation", 1.0)
        sharpness = params.get("sharpness", 1.0)
        filter_type = params.get("filter", "normal")
        
        # 创建原始图片的副本以避免修改原图
        img = original_image.copy()
        
        # 应用滤镜
        if filter_type != "normal":
            img = ImageProcessor.apply_filter(img, filter_type)
        
        # 应用亮度调整
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        # 应用对比度调整
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # 应用饱和度调整
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        
        # 应用清晰度调整
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)
        
        return img