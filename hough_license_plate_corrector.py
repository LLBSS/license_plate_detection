import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class EdgeBasedLicensePlateCorrector:
    """
    基于边缘检测的车牌矫正器
    """
    
    def correct_license_plate(self, image_path):
        """使用边缘检测和霍夫变换矫正车牌"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return None, False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 边缘检测
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            print("未检测到直线")
            return img, False
        
        # 计算所有直线的角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # 只考虑接近水平的直线
                angles.append(angle)
        
        if not angles:
            print("未找到合适的直线")
            return img, False
        
        # 计算平均角度
        avg_angle = np.mean(angles)
        print(f"检测到的倾斜角度: {avg_angle:.2f}°")
        
        # 旋转矫正
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        
        # 计算新边界
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, True


def resize_image(image, max_width=800, max_height=600):
    """
    调整图像大小，使其不超过指定的最大宽度和高度
    
    参数:
    image: 输入图像
    max_width: 最大宽度
    max_height: 最大高度
    
    返回:
    调整大小后的图像
    """
    height, width = image.shape[:2]
    
    # 计算缩放比例
    scale = min(max_width / width, max_height / height)
    
    # 如果不需要缩放，直接返回原图像
    if scale >= 1:
        return image
    
    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 调整大小
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


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


class LicensePlateApp:
    """
    车牌矫正可视化应用
    """
    def __init__(self, root):
        self.root = root
        self.root.title("车牌矫正")
        self.root.geometry("1000x700")
        
        # 设置test文件夹路径
        self.test_folder = "test"
        self.image_files = []
        self.current_image = None
        self.current_result = None
        
        # 创建矫正器实例
        self.corrector = EdgeBasedLicensePlateCorrector()
        
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
        title_label = ttk.Label(top_frame, text="车牌矫正", font=("Arial", 16))
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
        
        # 进行车牌矫正
        result, detected_lines = self.corrector.correct_license_plate(image_path)
        
        if result is not None:
            self.current_image = selected_file
            self.current_result = result
            
            # 调整展示大小
            resized_result = resize_image(result)
            
            # 显示结果
            self.display_image(resized_result)
            
            # 启用保存按钮
            self.save_button.config(state=tk.NORMAL)
            
            if detected_lines:
                # 检测到直线，自动保存结果
                output_name = f"{os.path.splitext(selected_file)[0]}_corrected{os.path.splitext(selected_file)[1]}"
                output_path = os.path.join(self.test_folder, output_name)
                if cv2.imwrite(output_path, result):
                    self.status_label.config(text=f"检测到直线，结果已自动保存: {output_name}")
                else:
                    self.status_label.config(text=f"检测到直线，但保存失败: {selected_file}")
            else:
                # 未检测到直线，删除原图像
                try:
                    os.remove(image_path)
                    # 从列表中移除
                    self.image_listbox.delete(index)
                    self.image_files.pop(index)
                    self.status_label.config(text=f"未检测到直线，原图像已删除: {selected_file}")
                except Exception as e:
                    self.status_label.config(text=f"删除失败: {str(e)}")
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
        output_name = f"{os.path.splitext(self.current_image)[0]}_corrected{os.path.splitext(self.current_image)[1]}"
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