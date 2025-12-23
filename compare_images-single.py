# coding=utf-8
import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np

class BinaryImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("四张二值化图像对比展示")
        # 设置窗口大小
        self.root.geometry("1400x900")
        # 初始化变量
        self.image_path = None
        self.original_image = None
        self.binary_images = {}
        # 二值化参数 - 为四个窗口分别设置参数
        self.params = {
            "二值化图像1": {
                "threshold_type": tk.StringVar(value="全局阈值"),
                "threshold_value": tk.IntVar(value=127),
                "adaptive_method": tk.StringVar(value="均值自适应"),
                "block_size": tk.IntVar(value=11),
                "c_value": tk.IntVar(value=2)
            },
            "二值化图像2": {
                "threshold_type": tk.StringVar(value="全局阈值"),
                "threshold_value": tk.IntVar(value=150),
                "adaptive_method": tk.StringVar(value="均值自适应"),
                "block_size": tk.IntVar(value=15),
                "c_value": tk.IntVar(value=3)
            },
            "二值化图像3": {
                "threshold_type": tk.StringVar(value="自适应阈值"),
                "threshold_value": tk.IntVar(value=100),
                "adaptive_method": tk.StringVar(value="高斯自适应"),
                "block_size": tk.IntVar(value=9),
                "c_value": tk.IntVar(value=2)
            },
            "二值化图像4": {
                "threshold_type": tk.StringVar(value="自适应阈值"),
                "threshold_value": tk.IntVar(value=180),
                "adaptive_method": tk.StringVar(value="高斯自适应"),
                "block_size": tk.IntVar(value=13),
                "c_value": tk.IntVar(value=4)
            }
        }
        self.create_widgets()
    def create_widgets(self):
        """
        创建GUI组件
        """
        # 顶部控制区域
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        # 选择图片按钮
        self.select_btn = ttk.Button(top_frame, text="选择图片", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        # 图片路径显示
        self.path_label = ttk.Label(top_frame, text="未选择图片")
        self.path_label.pack(side=tk.LEFT, padx=5)
        # 应用参数按钮
        self.apply_btn = ttk.Button(top_frame, text="应用所有参数", command=self.process_image)
        self.apply_btn.pack(side=tk.RIGHT, padx=5)
        # 参数控制区域（为每个二值化图像创建参数面板）
        params_frame = ttk.LabelFrame(self.root, text="二值化参数控制", padding="10")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        # 为每个二值化图像创建参数控制
        for i, (name, param) in enumerate(self.params.items()):
            frame = ttk.LabelFrame(params_frame, text=name, padding="5")
            frame.grid(row=i//2, column=i%2, padx=5, pady=5, sticky=tk.W+tk.E)
            # 阈值类型选择
            ttk.Label(frame, text="阈值类型:").grid(row=0, column=0, padx=5, pady=5)
            type_combo = ttk.Combobox(frame, textvariable=param["threshold_type"], values=["全局阈值", "自适应阈值"])
            type_combo.grid(row=0, column=1, padx=5, pady=5)
            # 全局阈值参数
            ttk.Label(frame, text="阈值:").grid(row=1, column=0, padx=5, pady=5)
            threshold_scale = ttk.Scale(frame, from_=0, to=255, variable=param["threshold_value"], orient=tk.HORIZONTAL)
            threshold_scale.grid(row=1, column=1, padx=5, pady=5)
            threshold_label = ttk.Label(frame, textvariable=param["threshold_value"])
            threshold_label.grid(row=1, column=2, padx=5, pady=5)
            # 自适应阈值参数
            ttk.Label(frame, text="自适应方法:").grid(row=2, column=0, padx=5, pady=5)
            adaptive_combo = ttk.Combobox(frame, textvariable=param["adaptive_method"], values=["均值自适应", "高斯自适应"])
            adaptive_combo.grid(row=2, column=1, padx=5, pady=5)
            ttk.Label(frame, text="块大小:").grid(row=3, column=0, padx=5, pady=5)
            block_scale = ttk.Scale(frame, from_=3, to=31, variable=param["block_size"], orient=tk.HORIZONTAL)
            block_scale.grid(row=3, column=1, padx=5, pady=5)
            block_label = ttk.Label(frame, textvariable=param["block_size"])
            block_label.grid(row=3, column=2, padx=5, pady=5)
            ttk.Label(frame, text="常量C:").grid(row=4, column=0, padx=5, pady=5)
            c_scale = ttk.Scale(frame, from_=-10, to=10, variable=param["c_value"], orient=tk.HORIZONTAL)
            c_scale.grid(row=4, column=1, padx=5, pady=5)
            c_label = ttk.Label(frame, textvariable=param["c_value"])
            c_label.grid(row=4, column=2, padx=5, pady=5)
        # 显示区域
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)
        # 创建四个并列的图片显示区域
        self.canvas_frames = {}
        self.image_canvases = {}
        self.image_names = list(self.params.keys())
        for i, name in enumerate(self.image_names):
            frame = ttk.LabelFrame(display_frame, text=name, padding="5")
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            # 创建画布用于显示图片
            canvas = tk.Canvas(frame, bg="lightgray")
            canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas_frames[name] = frame
            self.image_canvases[name] = canvas
    def select_image(self):
        """
        选择单张图片
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.path_label.config(text=os.path.basename(file_path))
            self.process_image()
    def process_image(self):
        """
        对图片进行二值化处理
        """
        if not self.image_path:
            return
        # 加载原始图片
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            return
        # 转换为灰度图
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # 为四个窗口分别进行二值化处理
        for name in self.image_names:
            param = self.params[name]
            threshold_type = param["threshold_type"].get()
            if threshold_type == "全局阈值":
                # 全局阈值处理
                threshold_value = param["threshold_value"].get()
                _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                # 自适应阈值处理
                adaptive_method = param["adaptive_method"].get()
                method = cv2.ADAPTIVE_THRESH_MEAN_C if adaptive_method == "均值自适应" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                block_size = param["block_size"].get()
                # 确保块大小为奇数
                if block_size % 2 == 0:
                    block_size += 1
                c_value = param["c_value"].get()
                binary_image = cv2.adaptiveThreshold(gray_image, 255, method, cv2.THRESH_BINARY, block_size, c_value)
            # 转换为BGR格式用于显示
            self.binary_images[name] = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        self.display_images()
    def convert_to_tk_image(self, cv_image, max_width=300, max_height=400):
        """
        将OpenCV图像转换为Tkinter可以显示的图像，并调整大小
        """
        # 调整图片大小以适应显示窗口
        h, w = cv_image.shape[:2]
        scale = min(max_width / w, max_height / h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
        # OpenCV使用BGR格式，而PIL使用RGB格式
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(pil_image)
    def display_images(self):
        """
        显示四张二值化图像
        """
        max_width = 300
        max_height = 400
        for name, canvas in self.image_canvases.items():
            # 清空画布
            canvas.delete("all")
            if name in self.binary_images:
                # 加载图片
                image = self.binary_images[name]
                # 将OpenCV图像转换为Tkinter图像
                tk_image = self.convert_to_tk_image(image, max_width, max_height)
                # 在画布上显示图片（居中显示）
                canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else max_width
                canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else max_height
                x = (canvas_width - tk_image.width()) // 2
                y = (canvas_height - tk_image.height()) // 2
                # 在画布上显示图片
                canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
                # 保存图像引用，防止被垃圾回收
                canvas.image = tk_image

def main():
    # 创建Tkinter应用程序
    root = tk.Tk()
    # 创建BinaryImageComparisonApp实例
    app = BinaryImageComparisonApp(root)
    # 运行应用程序
    root.mainloop()

if __name__ == "__main__":
    main()