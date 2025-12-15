import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageViewer:
    def __init__(self, title="图片展示器"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1000x850")
        
        # 创建主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建图片显示区域
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # 存储图片和标签
        self.images = []
        self.labels = []
        
    def display_images(self, result_dict):
        # 清空现有图片
        self.clear_images()
        # 处理车牌检测结果字典
        image_list = self.process_detection_results(result_dict)
        if(type(image_list) == list):
            # 计算布局
            rows, cols = self.calculate_layout(len(image_list))
            # 加载并显示图片
            for i, (image, title) in enumerate(image_list):
                try:
                    # 调整图片大小以适应窗口
                    max_width = 1000 // cols - 20
                    max_height = 1000 // rows - 20
                    
                    # 将numpy数组转换为PIL图像
                    if isinstance(image, np.ndarray):
                        # 将BGR转换为RGB（如果是OpenCV图像）
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(image)
                    else:
                        # 如果已经是PIL图像，直接使用
                        pil_image = image
                    
                    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    # 转换为Tkinter可用的格式
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # 创建标签显示图片（包含标题）
                    frame = tk.Frame(self.image_frame, relief=tk.RAISED, borderwidth=1)
                    frame.grid(row=i//cols, column=i%cols, padx=5, pady=5)
                    
                    label = tk.Label(frame, image=photo)
                    label.pack()
                    
                    # 添加标题
                    title_label = tk.Label(frame, text=title, font=("Arial", 10))
                    title_label.pack()
                    
                    label.image = photo  # 保持引用，防止被垃圾回收
                    
                    # 存储图片和标签
                    self.images.append(photo)
                    self.labels.append((frame, label, title_label))
                    
                except Exception as e:
                    messagebox.showerror("错误", f"无法加载图片: {title}\n错误信息: {str(e)}")
                  
    def process_detection_results(self, result_dict):
        """处理车牌检测结果字典"""
        image_list = []
        if isinstance(result_dict, np.ndarray):
            if result_dict.size > 0:
                image_list.append((result_dict, "处理结果"))
            return image_list
        # 添加原始图像（如果有）
        if 'original_image' in result_dict:
            image_list.append((result_dict['original_image'], "原始图像"))
        processing_steps = [
            ('gray_image', '灰度图像'),
            ('canny_edges', 'Canny边缘'),
            ('white_mask', '白色掩码'),
            ('opened_edges', '开运算'),
            ('closed_edges', '闭运算'),
            ('vertical_edges', '垂直边缘'),
            ('enhanced_vertical', '增强垂直边缘'),
            ('combined_result', '结合结果'),
            ('first_contours', '轮廓'),
            ('license_plates', '检测结果')
        ]
        
        for key, title in processing_steps:
            if key in result_dict and result_dict[key] is not None:
                if isinstance(result_dict[key], np.ndarray) and result_dict[key].size > 0:
                    image_list.append((result_dict[key], title))
        return image_list
    
    def draw_license_plates(self, result_dict):
        """在原图上绘制检测到的车牌区域"""
        try:
            result_image = result_dict['original_image'].copy()
            license_plates = result_dict.get('license_plates', [])
            
            for i, plate in enumerate(license_plates):
                # 假设plate是轮廓
                x, y, w, h = cv2.boundingRect(plate)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f'Plate {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return result_image
        except Exception as e:
            print(f"绘制车牌区域时出错: {e}")
            return None
    
    def calculate_layout(self, num_images):
        if num_images <= 0:
            return 0, 0
        elif num_images == 1:
            return 1, 1
        elif num_images == 2:
            return 1, 2
        elif num_images <= 4:
            return 2, 2
        elif num_images <= 6:
            return 2, 3
        elif num_images <= 9:
            return 3, 3
        else:
            return 3, 3
    
    def clear_images(self):
        """清空所有图片"""
        for label_info in self.labels:
            if isinstance(label_info, tuple):
                # 如果是元组，包含frame, label, title_label
                for widget in label_info:
                    widget.destroy()
        
        self.images.clear()
        self.labels.clear()
    
    def run(self):
        """运行应用程序"""
        self.root.mainloop()

def show_images(result_dict, title="车牌检测结果"):
    viewer = ImageViewer(title)
    viewer.display_images(result_dict)
    viewer.run()