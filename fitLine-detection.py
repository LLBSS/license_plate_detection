import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def draw_fit_lines(image_path):
    """
    在原图像上绘制使用cv2.fitLine拟合的上下边界线
    
    参数:
    image_path: 输入图片路径
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    original = img.copy()
    height, width = img.shape[:2]
    
    # 转换为灰度图并进行自适应二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 180, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到轮廓")
        return original
    
    # 找到最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    
    # 提取轮廓点
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    
    # 计算轮廓的中心Y坐标
    center_y = np.mean(contour_points[:, 1])
    
    # 分离上下边界点
    upper_points = contour_points[contour_points[:, 1] < center_y]
    lower_points = contour_points[contour_points[:, 1] > center_y]
    
    # 如果分离点太少，尝试另一种方法
    if len(upper_points) < 10 or len(lower_points) < 10:
        # 按Y坐标排序
        sorted_points_y = sorted(contour_points, key=lambda p: p[1])
        mid = len(sorted_points_y) // 2
        upper_points = np.array(sorted_points_y[:mid], dtype=np.float32)
        lower_points = np.array(sorted_points_y[mid:], dtype=np.float32)
    
    # 使用cv2.fitLine拟合上下边界线
    upper_line = None
    lower_line = None
    
    if len(upper_points) > 3:
        upper_line = cv2.fitLine(upper_points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    if len(lower_points) > 3:
        lower_line = cv2.fitLine(lower_points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 在原图上绘制上下边界线
    if upper_line is not None:
        # 提取直线参数
        vx_upper = upper_line[0][0]
        vy_upper = upper_line[1][0]
        x0_upper = upper_line[2][0]
        y0_upper = upper_line[3][0]
        
        # 计算直线在图像边界的两个点
        if abs(vx_upper) > abs(vy_upper):
            # 水平线为主
            x1 = 0
            y1 = int(y0_upper + (x1 - x0_upper) * (vy_upper / vx_upper))
            x2 = width - 1
            y2 = int(y0_upper + (x2 - x0_upper) * (vy_upper / vx_upper))
        else:
            # 垂直线为主
            y1 = 0
            x1 = int(x0_upper + (y1 - y0_upper) * (vx_upper / vy_upper))
            y2 = height - 1
            x2 = int(x0_upper + (y2 - y0_upper) * (vx_upper / vy_upper))
        
        # 绘制上边界线（红色）
        cv2.line(original, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    if lower_line is not None:
        # 提取直线参数
        vx_lower = lower_line[0][0]
        vy_lower = lower_line[1][0]
        x0_lower = lower_line[2][0]
        y0_lower = lower_line[3][0]
        
        # 计算直线在图像边界的两个点
        if abs(vx_lower) > abs(vy_lower):
            # 水平线为主
            x1 = 0
            y1 = int(y0_lower + (x1 - x0_lower) * (vy_lower / vx_lower))
            x2 = width - 1
            y2 = int(y0_lower + (x2 - x0_lower) * (vy_lower / vx_lower))
        else:
            # 垂直线为主
            y1 = 0
            x1 = int(x0_lower + (y1 - y0_lower) * (vx_lower / vy_lower))
            y2 = height - 1
            x2 = int(x0_lower + (y2 - y0_lower) * (vx_lower / vy_lower))
        
        # 绘制下边界线（蓝色）
        cv2.line(original, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return original


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
    车牌边界线拟合可视化应用
    """
    def __init__(self, root):
        self.root = root
        self.root.title("车牌边界线拟合")
        self.root.geometry("1000x700")
        
        # 设置test文件夹路径
        self.test_folder = "test"
        self.image_files = []
        self.current_image = None
        self.current_result = None
        
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
        title_label = ttk.Label(top_frame, text="车牌边界线拟合", font=("Arial", 16))
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
        
        # 绘制边界线
        result = draw_fit_lines(image_path)
        
        if result is not None:
            self.current_image = selected_file
            self.current_result = result
            
            # 调整展示大小
            resized_result = resize_image(result)
            
            # 显示结果
            self.display_image(resized_result)
            
            # 启用保存按钮
            self.save_button.config(state=tk.NORMAL)
            
            self.status_label.config(text=f"处理完成: {selected_file}")
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
        output_name = f"{os.path.splitext(self.current_image)[0]}_with_lines{os.path.splitext(self.current_image)[1]}"
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