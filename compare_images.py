# coding=utf-8
import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np

class ImageCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片并列展示对比")
        # 设置窗口大小
        self.root.geometry("1400x600")
        # 文件夹路径设置
        self.base_dir = os.getcwd()
        self.folders = {
            "原始图片": os.path.join(self.base_dir, "examples"),
            "颜色处理": os.path.join(self.base_dir, "examples-color"),
            "FLD线段": os.path.join(self.base_dir, "examples-color-fld")
        }
        # 获取所有图片文件名（基于原始图片文件夹）
        self.image_files = self.get_image_files(self.folders["原始图片"])
        self.current_index = 0
        self.create_widgets()
        self.display_images()
    def get_image_files(self, folder):
        """
        获取文件夹中的所有图片文件
        """
        image_files = []
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_files.append(filename)
        return sorted(image_files)
    def load_image(self, folder, filename, max_width=400, max_height=500):
        """
        加载图片并调整大小以适应显示
        """
        if folder == "FLD线段":
            # FLD文件夹中的文件名格式为 fld_原始文件名
            file_path = os.path.join(self.folders[folder], f"fld_{filename}")
        else:
            file_path = os.path.join(self.folders[folder], filename)
        if not os.path.exists(file_path):
            print(f"图片不存在: {file_path}")
            # 创建一个空白图像
            blank_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            blank_image[:] = (200, 200, 200)  # 灰色背景
            return blank_image
        # 使用OpenCV加载图片
        image = cv2.imread(file_path)
        if image is None:
            print(f"无法读取图片: {file_path}")
            # 创建一个空白图像
            blank_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            blank_image[:] = (200, 200, 200)  # 灰色背景
            return blank_image
        # 调整图片大小以适应显示窗口
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        return image
    def convert_to_tk_image(self, cv_image):
        """
        将OpenCV图像转换为Tkinter可以显示的图像
        """
        # OpenCV使用BGR格式，而PIL使用RGB格式
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(pil_image)
    
    def create_widgets(self):
        """
        创建GUI组件
        """
        # 顶部选择区域
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        # 图片选择标签和下拉菜单
        ttk.Label(top_frame, text="选择图片:", font=12).pack(side=tk.LEFT, padx=5)
        self.image_var = tk.StringVar()
        self.image_combobox = ttk.Combobox(top_frame, textvariable=self.image_var, values=self.image_files, width=30)
        self.image_combobox.pack(side=tk.LEFT, padx=5)
        self.image_combobox.bind("<<ComboboxSelected>>", self.on_image_selected)
        # 显示区域
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)
        # 创建三个并列的图片显示区域
        self.canvas_frames = {}
        self.image_canvases = {}
        for i, (name, folder) in enumerate(self.folders.items()):
            frame = ttk.LabelFrame(display_frame, text=name, padding="5")
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            # 创建画布用于显示图片
            canvas = tk.Canvas(frame, bg="lightgray")
            canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas_frames[name] = frame
            self.image_canvases[name] = canvas
    def display_images(self):
        """
        显示当前选中的图片
        """
        if not self.image_files:
            return
        # 获取当前选中的图片文件名
        filename = self.image_files[self.current_index]
        self.image_combobox.set(filename)
        max_width = 400
        max_height = 500
        for name, canvas in self.image_canvases.items():
            # 清空画布
            canvas.delete("all")
            # 加载图片
            image = self.load_image(name, filename, max_width, max_height)
            # 将OpenCV图像转换为Tkinter图像
            tk_image = self.convert_to_tk_image(image)
            # 在画布上显示图片（居中显示）
            canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else max_width
            canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else max_height
            x = (canvas_width - tk_image.width()) // 2
            y = (canvas_height - tk_image.height()) // 2
            # 在画布上显示图片
            canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
            # 保存图像引用，防止被垃圾回收
            canvas.image = tk_image
    def on_image_selected(self, event):
        """
        当用户选择新图片时更新显示
        """
        selected_file = self.image_var.get()
        if selected_file in self.image_files:
            self.current_index = self.image_files.index(selected_file)
            self.display_images()
    def update_canvas_sizes(self, event):
        """
        当窗口大小改变时更新画布上的图片位置
        """
        self.display_images()
def main():
    # 创建Tkinter应用程序
    root = tk.Tk()
    # 创建ImageCompareApp实例
    app = ImageCompareApp(root)
    # 绑定窗口大小改变事件
    root.bind("<Configure>", app.update_canvas_sizes)
    # 运行应用程序
    root.mainloop()
if __name__ == "__main__":
    main()