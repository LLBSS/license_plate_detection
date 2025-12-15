import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import cv2

# 导入各个模块
from python_GUI.image_processing import ImageProcessor
from python_GUI.image_display import ImageDisplayManager
from python_GUI.ui_manager import UIManager
from python_GUI.preset_manager import PresetManager
from python_GUI.event_handler import EventHandler
from detectors import LicensePlateDetector

class ImageEditorApp:
    """图像编辑器主应用类，整合所有功能模块"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("车牌检测工具")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 初始化图片数据
        self.original_images = [None, None]  # 原图
        self.processed_images = [None, None]  # 处理后的图像
        self.image_tk = [None, None]  # Tkinter显示图像
        self.selected_image_index = 0  # 当前选中的图片索引
        
        # 车牌检测结果
        self.detection_results = [None, None]
        
        # 初始化处理参数
        self.params = {
            "brightness": [1.0, 1.0],
            "contrast": [1.0, 1.0],
            "saturation": [1.0, 1.0],
            "sharpness": [1.0, 1.0],
            "zoom": [1.0, 1.0],
            "filter": ["gray_image", "gray_image"]
        }
        
        # 创建车牌检测器
        self.license_detector = LicensePlateDetector()
        
        # 创建并初始化各个模块
        self._initialize_modules()
        
        # 设置UI
        self.ui_manager.create_main_layout()
        
        # 设置事件绑定
        self.event_handler.setup_event_bindings()
        
        # 设置样式
        self._setup_styles()
    
    def _initialize_modules(self):
        """初始化各个功能模块"""
        self.image_processor = ImageProcessor()
        self.display_manager = ImageDisplayManager(self)
        self.ui_manager = UIManager(self)
        self.preset_manager = PresetManager(self)
        self.event_handler = EventHandler(self)
        self.license_plate_detector = LicensePlateDetector()
    
    def _setup_styles(self):
        """设置UI样式"""
        style = ttk.Style()
        # 创建选中状态的框架样式
        style.configure("Selected.TFrame", background="#e0e0ff")
        # 设置默认字体
        style.configure(".", font=("SimHei", 9))
    
    def open_image(self, index):
        """打开图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.ico *.webp"),
                      ("所有文件", "*.*")]
        )
        
        if file_path:
            self._load_image(file_path, index)
    
    def _load_image(self, file_path, index):
        """加载图片并显示"""
        try:
            # 显示加载状态
            self.ui_manager.update_status(f"正在加载图片...", timeout=0)
            self.root.update_idletasks()
            
            # 打开图片
            image = Image.open(file_path)
            # 确保图片模式兼容
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 保存原图
            self.original_images[index] = image
            
            # 清除之前的检测结果
            self.detection_results[index] = None
            
            # 更新图片标题
            file_name = os.path.basename(file_path)
            self.ui_manager.image_titles[index].config(text=f"图片 {index + 1}: {file_name}")
            
            # 更新选中状态
            if self.selected_image_index == index:
                self.ui_manager.select_image(index)
            
            # 应用当前参数处理图片并显示
            self.process_image(index)
            
            # 更新状态
            self.ui_manager.update_status(f"已加载图片: {file_name}")
            
        except Exception as e:
            self.ui_manager.show_error(f"加载图片失败: {str(e)}")
            self.ui_manager.update_status("就绪")
    
    def process_image(self, index):
        """处理图片并显示图像"""
        if self.original_images[index] is None:
            return
        
        try:
            # 获取当前选择的结果类型
            result_type = self.params["filter"][index]
            
            # 将PIL图像转换为OpenCV格式
            pil_image = self.original_images[index].copy()
            # PIL (RGB) -> OpenCV (BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 执行车牌检测
            if self.detection_results[index] is None:
                self.detection_results[index] = self.license_detector.detect_license_plates(cv_image)
                self.ui_manager.update_status(f"已完成车牌检测")
            
            # 获取选择的检测结果
            if result_type in self.detection_results[index]:
                result_image = self.detection_results[index][result_type]
                
                # 如果结果是单通道图像，转换为三通道以显示
                if len(result_image.shape) == 2:
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
                else:
                    # 如果是BGR格式，转换为RGB
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # 转换回PIL格式
                self.processed_images[index] = Image.fromarray(result_image)
            else:
                # 如果结果类型不存在，显示原图
                self.processed_images[index] = self.original_images[index].copy()
            
            # 显示图片
            self.display_manager.display_image(index)
            
        except Exception as e:
            self.ui_manager.show_error(f"处理图片失败: {str(e)}")
    
    def on_param_change(self, param, value):
        """参数变化处理"""
        idx = self.selected_image_index
        value = float(value)
        
        # 更新参数值
        self.params[param][idx] = value
        
        # 更新UI显示
        self.param_controls[f"{param}_value"].config(text=f"{value:.1f}")
        
        # 如果是缩放参数，直接更新显示
        if param == "zoom":
            self.display_manager.display_image(idx)
        else:
            # 其他参数需要重新处理图片
            self.process_image(idx)
    
    def adjust_param(self, param, delta):
        """微调参数"""
        idx = self.selected_image_index
        current_value = self.params[param][idx]
        new_value = max(0.1, min(5.0, current_value + delta))
        
        # 更新参数值
        self.params[param][idx] = new_value
        
        # 更新UI控件
        self.param_controls[param].set(new_value)
        self.param_controls[f"{param}_value"].config(text=f"{new_value:.1f}")
        
        # 应用更改
        if param == "zoom":
            self.display_manager.display_image(idx)
        else:
            self.process_image(idx)
    
    def reset_params(self):
        """重置参数"""
        idx = self.selected_image_index
        
        # 重置参数
        self.params["brightness"][idx] = 1.0
        self.params["contrast"][idx] = 1.0
        self.params["saturation"][idx] = 1.0
        self.params["sharpness"][idx] = 1.0
        self.params["zoom"][idx] = 1.0
        self.params["filter"][idx] = "gray_image"
        
        # 清除检测结果，重新检测
        self.detection_results[idx] = None
        
        # 更新UI控件显示
        self.ui_manager.update_param_controls()
        
        # 重新处理图片
        self.process_image(idx)
        
        # 更新状态
        self.ui_manager.update_status("参数已重置")
    
    def on_filter_change(self, event=None):
        """车牌检测结果类型选择处理"""
        result_type = self.filter_var.get()
        idx = self.selected_image_index
        
        # 更新参数
        self.params["filter"][idx] = result_type
        
        # 重新处理图片
        self.process_image(idx)
        
        # 更新状态
        self.ui_manager.update_status(f"已选择检测结果类型: {result_type}")
    
    def on_compare_mode_change(self):
        """对比模式变化处理"""
        # 更新图片显示
        for i in range(2):
            if self.original_images[i] is not None:
                self.display_manager.display_image(i)
    
    def select_image(self, index):
        """选择图片"""
        # 检查索引是否有效
        if 0 <= index < len(self.original_images):
            # 更新选中的图片索引
            self.selected_image_index = index
            # 更新UI状态
            self.ui_manager.update_status(f"已选择图片 {index + 1}")
            # 更新参数控件显示当前选中图片的参数
            self.ui_manager.update_param_controls()
    
    def on_window_resize(self, event=None):
        """窗口大小调整处理"""
        # 检查是否是由调整窗口大小引起的事件
        if event and hasattr(event, "widget") and event.widget == self.root:
            # 重新显示所有已加载的图片
            for i in range(2):
                if self.original_images[i] is not None:
                    self.display_manager.display_image(i)
    
    def save_image(self):
        """保存图片"""
        idx = self.selected_image_index
        if self.processed_images[idx] is None:
            self.ui_manager.show_error("没有可保存的图片")
            return
        
        # 获取原始图片格式
        original_format = "JPEG"
        if self.original_images[idx] is not None:
            original_format = self.original_images[idx].format or "JPEG"
        
        # 文件保存对话框
        file_path = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension="." + original_format.lower(),
            filetypes=[
                ("JPEG 图片", "*.jpg"),
                ("PNG 图片", "*.png"),
                ("BMP 图片", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 保存图片
                self.processed_images[idx].save(file_path)
                self.ui_manager.update_status(f"图片已保存: {file_path}")
            except Exception as e:
                self.ui_manager.show_error(f"保存图片失败: {str(e)}")
    
    def clear_current_image(self):
        """清空当前选中的图片"""
        idx = self.selected_image_index
        
        # 清空图片数据
        self.original_images[idx] = None
        self.processed_images[idx] = None
        self.image_tk[idx] = None
        
        # 清空画布
        self.ui_manager.image_labels[idx].delete("all")
        self.ui_manager.image_titles[idx].config(text=f"图片 {idx + 1}")
        
        # 重置参数
        self.params["brightness"][idx] = 1.0
        self.params["contrast"][idx] = 1.0
        self.params["saturation"][idx] = 1.0
        self.params["sharpness"][idx] = 1.0
        self.params["zoom"][idx] = 1.0
        self.params["filter"][idx] = "gray_image"
        
        # 更新UI
        if self.selected_image_index == idx:
            self.ui_manager.update_param_controls()
            self.ui_manager.filter_var.set("gray_image")
        
        # 更新状态
        self.ui_manager.update_status(f"已清空图片 {idx + 1}")
    def process_license_plate(self, index = 0):
        """处理车牌图片"""
        idx = index
        # 检查图片是否已加载
        if self.original_images[idx] is None:
            self.ui_manager.show_error("请先加载图片")
            return
        
        try:
            # 调用车牌识别函数
            original_image = np.array(self.original_images[idx])
            detection_result = self.license_plate_detector.detect_license_plates(original_image)
            
            # 获取识别结果
            combined_result = detection_result['combined_result']
            license_plates = detection_result['license_plates']
            
            # 转换回 PIL 图像
            pil_image = Image.fromarray(combined_result)
            self.processed_images[1-idx] = pil_image
            
            # 调用图片展示函数
            self.display_manager.display_image(1-idx)
            
            # 更新状态
            if len(license_plates) > 0:
                self.ui_manager.update_status(f"识别到{len(license_plates)}个车牌")
            else:
                self.ui_manager.update_status("未识别到车牌")
        except Exception as e:
            self.ui_manager.show_error(f"车牌识别失败: {str(e)}")

    def show_help(self):
        """显示帮助信息"""
        self.ui_manager.show_help()
    
    def show_error(self, title, message):
        """显示错误信息"""
        messagebox.showerror(title, message)
    
    def show_info(self, title, message):
        """显示信息"""
        messagebox.showinfo(title, message)
    
    def on_window_close(self):
        """窗口关闭处理"""
        # 释放资源
        for i in range(2):
            self.original_images[i] = None
            self.processed_images[i] = None
            self.image_tk[i] = None
        
        # 关闭窗口
        self.root.destroy()
    
    def __del__(self):
        """析构函数，清理资源"""
        for i in range(2):
            self.original_images[i] = None
            self.processed_images[i] = None
            self.image_tk[i] = None

# 主函数
if __name__ == "__main__":
    root = tk.Tk()
    # 设置中文字体支持
    app = ImageEditorApp(root)
    root.mainloop()