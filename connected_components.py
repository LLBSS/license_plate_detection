# coding=utf-8
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

class ConnectedComponentsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("连通组件检测与标记")
        # 设置窗口大小
        self.root.geometry("1000x600")
        # 初始化变量
        self.image_path = None
        self.original_image = None
        self.labeled_image = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """
        创建GUI组件
        """
        # 顶部控制区域
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        # 选择图片按钮
        self.select_btn = ttk.Button(top_frame, text="选择二值化图片", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # 图片路径显示
        self.path_label = ttk.Label(top_frame, text="未选择图片")
        self.path_label.pack(side=tk.LEFT, padx=5)
        
        # 应用处理按钮
        self.process_btn = ttk.Button(top_frame, text="检测连通组件", command=self.process_image)
        self.process_btn.pack(side=tk.RIGHT, padx=5)
        
        # 显示区域
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建两个并列的图片显示区域
        self.canvas_frames = {}
        self.image_canvases = {}
        self.image_names = ["原图", "标记后的图片"]
        
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
        选择二值化图片
        """
        file_path = filedialog.askopenfilename(
            initialdir="examples-binary",  # 设置默认路径
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.path_label.config(text=os.path.basename(file_path))
            self.load_and_display_original()
    
    def load_and_display_original(self):
        """
        加载并显示原始图片
        """
        if not self.image_path:
            return
        
        # 加载原始图片
        self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            return
        
        # 将灰度图转换为BGR格式用于显示
        bgr_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        self.display_image(bgr_image, "原图")
    
    def save_component_info(self, image_path, num_labels, labels, stats, centroids, area_threshold, total_area):
        """
        保存筛选后的连通组件信息到txt文件
        """
        # 创建test-txt文件夹（如果不存在）
        os.makedirs("test-txt", exist_ok=True)
        
        # 提取输入图像的文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join("test-txt", f"{filename}.txt")
        
        # 收集符合条件的连通组件信息
        component_info = []
        for i in range(num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 跳过背景和面积小于阈值的组件
            if i == 0 or area < area_threshold:
                continue  
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx = centroids[i, 0]
            cy = centroids[i, 1]
            area_ratio = (area / total_area) * 100
            
            # 计算填充率并进行第二次判断
            bounding_area = w * h
            fill_ratio = (area / bounding_area) * 100
            if fill_ratio >= 15:
                component_info.append({
                    'id': i,
                    'area': area,
                    'area_ratio': area_ratio,
                    'fill_ratio': fill_ratio,
                    'bounding_box': (x, y, w, h),
                    'centroid': (cx, cy)
                })
        
        # 写入txt文件
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("筛选后的连通组件信息（面积≥0.2%且填充率≥15%）\n")
            f.write("=" * 80 + "\n")
            f.write(f"图像文件名：{os.path.basename(image_path)}\n")
            f.write(f"总筛选后的组件数量：{len(component_info)}\n")
            f.write(f"图像总面积：{total_area} 像素\n")
            f.write(f"面积阈值（0.2%）：{area_threshold:.0f} 像素\n")
            f.write("=" * 80 + "\n")
            f.write("组件ID | 面积(像素) | 占比(%) | 填充率(%) | 边界框(x, y, w, h) | 质心坐标(cx, cy)\n")
            f.write("-" * 105 + "\n")
            for info in component_info:
                f.write(f"{info['id']:4d} | {info['area']:10d} | {info['area_ratio']:5.1f}% | {info['fill_ratio']:5.1f}% | ({info['bounding_box'][0]:4d}, {info['bounding_box'][1]:4d}, {info['bounding_box'][2]:4d}, {info['bounding_box'][3]:4d}) | ({info['centroid'][0]:6.1f}, {info['centroid'][1]:6.1f})\n")
            f.write("-" * 105 + "\n")
        
        print(f"\n连通组件信息已保存到：{txt_path}")
    
    def process_image(self):
        """
        检测连通组件并进行标记
        """
        if not self.image_path or self.original_image is None:
            return
        
        # 确保图像是二值化的
        _, binary_image = cv2.threshold(self.original_image, 127, 255, cv2.THRESH_BINARY)
        
        # 检测连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)
        
        # 计算图像总面积和0.2%阈值
        image_height, image_width = binary_image.shape
        total_area = image_height * image_width
        area_threshold = total_area * 0.001 # 0.2%的面积阈值
        # 统计符合条件的物体数量
        valid_objects = 0
        for i in range(1, num_labels):  # 从1开始，不包括背景
            # 第一次判断：面积大于0.2%
            if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                # 第二次判断：标记的图像面积占连通矩阵面积的15%以上
                component_area = stats[i, cv2.CC_STAT_AREA]
                bounding_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
                if component_area / bounding_area >= 0.15:
                    valid_objects += 1
        
        # 打印连通组件检测结果
        print("连通组件检测结果：")
        print(f"总连通组件数量（包括背景）：{num_labels}")
        print(f"实际物体数量（不包括背景）：{num_labels - 1}")
        print(f"面积大于0.2%的物体数量：{valid_objects}")
        print(f"图像总面积：{total_area} 像素")
        print(f"面积阈值（0.2%）：{area_threshold:.0f} 像素")
        
        # 打印符合条件的连通组件信息
        print("\n面积大于0.2%且填充率大于15%的连通组件信息：")
        print("组件ID | 面积(像素) | 占比(%) | 填充率(%) | 边界框(x, y, w, h) | 质心坐标(cx, cy)")
        print("-" * 105)
        for i in range(num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 跳过背景和面积小于阈值的组件
            if i == 0 or area < area_threshold:
                continue  
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx = centroids[i, 0]
            cy = centroids[i, 1]
            area_ratio = (area / total_area) * 100
            
            # 计算填充率并进行第二次判断
            bounding_area = w * h
            fill_ratio = (area / bounding_area) * 100
            if fill_ratio >= 15:
                print(f"{i:4d} | {area:10d} | {area_ratio:5.1f}% | {fill_ratio:5.1f}% | ({x:4d}, {y:4d}, {w:4d}, {h:4d}) | ({cx:6.1f}, {cy:6.1f})")
        print("-" * 105)
        
        # 保存连通组件信息到txt文件
        self.save_component_info(self.image_path, num_labels, labels, stats, centroids, area_threshold, total_area)
        
        # 生成随机颜色用于标记不同的连通组  要改
        colors = []
        for i in range(num_labels):
            # 背景用黑色
            if i == 0:
                colors.append([0, 0, 0])
            else:
                # 第一次判断：面积大于0.2%
                if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                    # 第二次判断：填充率大于15%
                    component_area = stats[i, cv2.CC_STAT_AREA]
                    bounding_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
                    if component_area / bounding_area >= 0.15:
                        colors.append([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
                    else:
                        colors.append([0, 0, 0])  # 使用背景色（不标记）
                else:
                    colors.append([0, 0, 0])  # 使用背景色（不标记）
        colors = np.array(colors, dtype=np.uint8)
        
        # 创建标记后的彩色图像
        labeled_colors = colors[labels]
        
        # 在标记后的图像上绘制组件边界（仅对面积大于0.2%且填充率大于15%的组件）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= area_threshold:
                # 第二次判断：填充率大于15%
                component_area = stats[i, cv2.CC_STAT_AREA]
                bounding_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
                if component_area / bounding_area >= 0.15:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    cv2.rectangle(labeled_colors, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # 保存标记后的图像
        self.labeled_image = labeled_colors
        
        # 显示标记后的图像
        self.display_image(labeled_colors, "标记后的图片")
    
    def display_image(self, cv_image, canvas_name):
        """
        在指定的画布上显示图片
        """
        if canvas_name not in self.image_canvases:
            return
        
        canvas = self.image_canvases[canvas_name]
        
        # 调整图片大小以适应显示窗口
        max_width = 450
        max_height = 500
        h, w = cv_image.shape[:2]
        scale = min(max_width / w, max_height / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
        
        # OpenCV使用BGR格式，而PIL使用RGB格式
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # 清空画布
        canvas.delete("all")
        
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
    # 创建ConnectedComponentsApp实例
    app = ConnectedComponentsApp(root)
    # 运行应用程序
    root.mainloop()

if __name__ == "__main__":
    main()