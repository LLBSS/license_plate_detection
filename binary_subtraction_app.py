# coding=utf-8
import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np

class BinarySubtractionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像二值化与相减展示")
        # 设置窗口大小
        self.root.geometry("1800x1200")
        # 初始化变量
        self.image_path = None
        self.original_image = None
        self.processed_images = {}
        # 二值化参数 - 只保留一个二值化处理的参数
        self.params = {
            "二值化图像": {
                "threshold_value": tk.IntVar(value=127),
                "reverse": tk.BooleanVar(value=False)
            },
            "侵蚀": {
                "kernel_size": tk.IntVar(value=3),
                "kernel_shape": tk.StringVar(value="矩形")
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
        self.apply_btn = ttk.Button(top_frame, text="应用参数", command=self.process_image)
        self.apply_btn.pack(side=tk.RIGHT, padx=5)
        # 保存结果按钮
        self.save_btn = ttk.Button(top_frame, text="保存结果", command=self.save_result)
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        # 参数控制区域
        params_frame = ttk.LabelFrame(self.root, text="二值化参数控制", padding="10")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        # 为二值化处理创建参数控制
        name = "二值化图像"
        param = self.params[name]
        frame = ttk.LabelFrame(params_frame, text=name, padding="5")
        frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        # 全局阈值参数
        ttk.Label(frame, text="阈值:").grid(row=0, column=0, padx=5, pady=5)
        threshold_scale = ttk.Scale(frame, from_=0, to=255, variable=param["threshold_value"], orient=tk.HORIZONTAL)
        threshold_scale.grid(row=0, column=1, padx=5, pady=5)
        threshold_label = ttk.Label(frame, textvariable=param["threshold_value"])
        threshold_label.grid(row=0, column=2, padx=5, pady=5)
        # 反转控制
        reverse_check = ttk.Checkbutton(frame, text="反转二值化结果", variable=param["reverse"])
        reverse_check.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        # 为侵蚀运算创建参数控制
        erosion_param = self.params["侵蚀"]
        erosion_frame = ttk.LabelFrame(params_frame, text="侵蚀参数", padding="5")
        erosion_frame.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        # 核大小参数
        ttk.Label(erosion_frame, text="核大小:").grid(row=0, column=0, padx=5, pady=5)
        kernel_scale = ttk.Scale(erosion_frame, from_=1, to=21, variable=erosion_param["kernel_size"], orient=tk.HORIZONTAL)
        kernel_scale.grid(row=0, column=1, padx=5, pady=5)
        kernel_label = ttk.Label(erosion_frame, textvariable=erosion_param["kernel_size"])
        kernel_label.grid(row=0, column=2, padx=5, pady=5)
        # 核形状参数
        ttk.Label(erosion_frame, text="核形状:").grid(row=1, column=0, padx=5, pady=5)
        shape_combobox = ttk.Combobox(erosion_frame, textvariable=erosion_param["kernel_shape"], values=["矩形", "十字形", "椭圆形"])
        shape_combobox.grid(row=1, column=1, padx=5, pady=5)
        shape_combobox.config(state="readonly")
        # 显示区域
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)
        # 创建三个并列的图片显示区域
        self.canvas_frames = {}
        self.image_canvases = {}
        self.image_names = ["原图", "二值化图像", "侵蚀结果"]
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
        对图片进行处理：二值化和相减
        """
        if not self.image_path:
            return
        # 加载原始图片
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            return
        # 保存原图到处理结果字典
        self.processed_images["原图"] = self.original_image
        # 转换为灰度图
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # 对二值化图像进行处理
        name = "二值化图像"
        param = self.params[name]
        # 全局阈值处理
        threshold_value = param["threshold_value"].get()
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        # 应用反转功能
        if param["reverse"].get():
            binary_image = cv2.bitwise_not(binary_image)
        # 转换为BGR格式用于显示
        self.processed_images[name] = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # 对二值化图像进行侵蚀运算
        self.perform_erosion()
        # 显示所有图像
        self.display_images()
    def perform_erosion(self):
        """
        对二值化图像执行侵蚀运算
        """
        if "二值化图像" in self.processed_images:
            # 获取侵蚀参数
            erosion_param = self.params["侵蚀"]
            kernel_size = erosion_param["kernel_size"].get()
            kernel_shape = erosion_param["kernel_shape"].get()
            # 获取二值化图像的灰度图
            binary_gray = cv2.cvtColor(self.processed_images["二值化图像"], cv2.COLOR_BGR2GRAY)
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
            # 转换为BGR格式用于显示
            self.processed_images["侵蚀结果"] = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
    def convert_to_tk_image(self, cv_image, max_width=350, max_height=450):
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
        显示四张图像
        """
        max_width = 350
        max_height = 450
        for name, canvas in self.image_canvases.items():
            # 清空画布
            canvas.delete("all")
            if name in self.processed_images:
                # 加载图片
                image = self.processed_images[name]
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
    def save_result(self):
        """
        保存最终的侵蚀结果图像
        """
        if "侵蚀结果" in self.processed_images:
            # 打开文件保存对话框
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp")],
                title="保存侵蚀结果图像"
            )
            if save_path:
                # 保存图像
                cv2.imwrite(save_path, self.processed_images["侵蚀结果"])

def main():
    # 创建Tkinter应用程序
    root = tk.Tk()
    # 创建BinarySubtractionApp实例
    app = BinarySubtractionApp(root)
    # 运行应用程序
    root.mainloop()

if __name__ == "__main__":
    main()