# coding=utf-8
import cv2
import os
import numpy as np
#图像全部二值化
class BinarySubtractionApp:
    def __init__(self):
        # 二值化参数
        self.params = {
            "二值化图像": {
                "threshold_value": 180,  # 默认阈值
                "reverse": False
            },
            "侵蚀": {
                "kernel_size": 3,
                "kernel_shape": "矩形"
            }
        }
        # 加载亮度阈值结果
        self.brightness_thresholds = self.load_brightness_thresholds()
        # 确保保存目录存在
        self.save_dir = "examples-binary"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def load_brightness_thresholds(self):
        """
        从brightness_threshold_batch_result.txt加载亮度阈值
        :return: 包含文件名和对应阈值的字典
        """
        thresholds = {}
        result_file = "brightness_threshold_batch_result.txt"
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 跳过前3行标题
                for line in lines[3:]:
                    line = line.strip()
                    if line:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            filename = parts[0]
                            threshold_90 = int(parts[1])
                            thresholds[filename] = threshold_90 +15
        except FileNotFoundError:
            print(f"警告：未找到阈值文件 {result_file}")
        except Exception as e:
            print(f"加载阈值文件时出错：{e}")
        return thresholds
    
    def process_image(self, image_path):
        """
        对图片进行处理：二值化和侵蚀
        """
        # 加载原始图片
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"无法读取图片：{image_path}")
            return
        
        filename = os.path.basename(image_path)
        
        # 检查是否有对应的亮度阈值
        if filename in self.brightness_thresholds:
            self.params["二值化图像"]["threshold_value"] = self.brightness_thresholds[filename]
            print(f"已使用自动计算的阈值：{self.params['二值化图像']['threshold_value']} (来自{filename})")
        else:
            # 使用默认阈值200
            print(f"未找到{filename}的阈值，使用默认阈值200")
        
        processed_images = {}
        processed_images["原图"] = original_image
        
        # 转换为灰度图
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 对二值化图像进行处理
        threshold_value = self.params["二值化图像"]["threshold_value"]
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 应用反转功能
        if self.params["二值化图像"]["reverse"]:
            binary_image = cv2.bitwise_not(binary_image)
        
        # 转换为BGR格式
        processed_images["二值化图像"] = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        
        # 对二值化图像进行侵蚀运算
        self.perform_erosion(processed_images)
        
        # 保存结果
        self.save_result(processed_images, filename)
    
    def perform_erosion(self, processed_images):
        """
        对二值化图像执行侵蚀运算
        """
        if "二值化图像" in processed_images:
            # 获取侵蚀参数
            kernel_size = self.params["侵蚀"]["kernel_size"]
            kernel_shape = self.params["侵蚀"]["kernel_shape"]
            
            # 获取二值化图像的灰度图
            binary_gray = cv2.cvtColor(processed_images["二值化图像"], cv2.COLOR_BGR2GRAY)
            
            # 创建侵蚀核
            if kernel_shape == "矩形":
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            elif kernel_shape == "十字形":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            elif kernel_shape == "椭圆形":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # 执行侵蚀运算
            eroded_image = cv2.erode(binary_gray, kernel, iterations=1)
            
            # 转换为BGR格式
            processed_images["侵蚀结果"] = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
    
    def save_result(self, processed_images, filename):
        """
        保存最终的侵蚀结果图像
        """
        if "侵蚀结果" in processed_images:
            # 构造保存路径
            base_name, ext = os.path.splitext(filename)
            save_name = f"{base_name}_binary{ext}"
            final_save_path = os.path.join(self.save_dir, save_name)
            
            # 保存图像
            cv2.imwrite(final_save_path, processed_images["侵蚀结果"])
            print(f"图像已保存到：{final_save_path}")
    
    def process_all_images(self, examples_dir):
        """
        处理examples文件夹中的所有图片
        """
        if not os.path.exists(examples_dir):
            print(f"错误：文件夹 {examples_dir} 不存在")
            return
        
        # 获取文件夹中的所有图片
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(examples_dir):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(examples_dir, file))
        
        if not image_files:
            print(f"在 {examples_dir} 中未找到图片")
            return
        
        # 处理所有图片
        print(f"开始处理 {len(image_files)} 张图片...")
        for image_file in image_files:
            print(f"\n处理图片：{os.path.basename(image_file)}")
            self.process_image(image_file)
        
        print(f"\n所有图片处理完成！结果保存在 {self.save_dir} 文件夹中")


def main():
    # 创建BinarySubtractionApp实例
    app = BinarySubtractionApp()
    
    # 处理examples文件夹中的所有图片
    examples_dir = "examples"
    app.process_all_images(examples_dir)


if __name__ == "__main__":
    main()