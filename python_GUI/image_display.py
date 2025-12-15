import tkinter as tk
from PIL import Image, ImageTk

class ImageDisplayManager:
    """图像显示管理类，负责在画布上显示图片，处理缩放、拖动和滑动对比等功能"""
    
    def __init__(self, app):
        self.app = app
        self.is_dragging = False
        self.is_dragging_line = False
        self.last_x = 0
        self.last_y = 0
        self.drag_line_pos = 0.5  # 滑动对比时的分割线位置
    
    def display_image(self, index):
        """在画布上显示图片"""
        if self.app.processed_images[index] is None:
            return
        
        try:
            # 获取画布大小
            canvas = self.app.image_labels[index]
            canvas_width = canvas.winfo_width() - 20  # 减去边距
            canvas_height = canvas.winfo_height() - 20
            
            # 如果画布还未渲染，则使用默认大小
            if canvas_width <= 10:  # 避免宽度过小
                canvas_width = 400
            if canvas_height <= 10:
                canvas_height = 300
            
            # 应用缩放
            zoom = self.app.params["zoom"][index]
            
            # 获取当前对比模式
            mode = self.app.compare_mode.get()
            
            # 如果是第一个画布且是滑动对比模式，需要同时使用原图和处理图
            if index == 0 and mode.startswith("slider_"):
                self._display_slider_comparison(index, canvas, canvas_width, canvas_height, zoom, mode)
            elif index == 0 and mode == "original":
                self._display_original_only(index, canvas, canvas_width, canvas_height, zoom)
            else:
                self._display_processed_image(index, canvas, canvas_width, canvas_height, zoom)
            
            # 更新滚动区域
            canvas.config(scrollregion=canvas.bbox("all"))
            
        except ValueError as e:
            self.app.show_error("显示错误", str(e))
        except Exception as e:
            self.app.show_error("显示失败", f"显示图片时出错:\n{str(e)}")
    
    def _display_slider_comparison(self, index, canvas, canvas_width, canvas_height, zoom, mode):
        """显示滑动对比效果"""
        # 复制图片以避免修改原始图片
        original_img = self.app.original_images[index].copy()
        processed_img = self.app.processed_images[index].copy()
        
        # 计算应用缩放后的尺寸
        orig_width, orig_height = processed_img.size
        new_width = int(orig_width * zoom)
        new_height = int(orig_height * zoom)
        
        # 检查计算后的尺寸是否有效
        if new_width <= 0 or new_height <= 0:
            raise ValueError("计算后的图片尺寸无效")
        
        # 计算图片在画布上的缩放比例以适应显示
        scale_width = canvas_width / new_width
        scale_height = canvas_height / new_height
        scale = min(scale_width, scale_height, 1.0)  # 不超过原始大小
        
        # 计算最终显示尺寸
        display_width = int(new_width * scale)
        display_height = int(new_height * scale)
        
        # 调整原图和处理图大小
        resized_original = original_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        resized_processed = processed_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # 清空画布
        canvas.delete("all")
        
        if mode == "slider_horizontal":
            self._display_horizontal_slider(canvas, canvas_width, canvas_height, 
                                           resized_original, resized_processed, 
                                           display_width, display_height)
        elif mode == "slider_vertical":
            self._display_vertical_slider(canvas, canvas_width, canvas_height, 
                                         resized_original, resized_processed, 
                                         display_width, display_height)
        
        # 绑定分割线拖动事件
        canvas.bind("<Button-1>", self.on_slider_line_click)
        canvas.bind("<B1-Motion>", self.on_slider_line_drag)
    
    def _display_horizontal_slider(self, canvas, canvas_width, canvas_height, 
                                  resized_original, resized_processed, 
                                  display_width, display_height):
        """显示水平滑动对比"""
        split_x = int(display_width * self.drag_line_pos)
        
        # 创建复合图像
        combined_img = Image.new('RGB', (display_width, display_height))
        # 左侧显示原图
        combined_img.paste(resized_original.crop((0, 0, split_x, display_height)), (0, 0))
        # 右侧显示处理图
        combined_img.paste(resized_processed.crop((split_x, 0, display_width, display_height)), (split_x, 0))
        
        # 转换为Tkinter可用的图像格式
        self.app.image_tk[0] = ImageTk.PhotoImage(combined_img)
        
        # 显示图片
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.app.image_tk[0])
        
        # 计算分割线位置
        line_x = (canvas_width - display_width) // 2 + split_x
        
        # 绘制分割线
        canvas.create_line(line_x, (canvas_height - display_height) // 2, 
                          line_x, (canvas_height + display_height) // 2, 
                          width=3, fill="white", dash=(5, 5))
    
    def _display_vertical_slider(self, canvas, canvas_width, canvas_height, 
                                resized_original, resized_processed, 
                                display_width, display_height):
        """显示垂直滑动对比"""
        split_y = int(display_height * self.drag_line_pos)
        
        # 创建复合图像
        combined_img = Image.new('RGB', (display_width, display_height))
        # 上部分显示原图
        combined_img.paste(resized_original.crop((0, 0, display_width, split_y)), (0, 0))
        # 下部分显示处理图
        combined_img.paste(resized_processed.crop((0, split_y, display_width, display_height)), (0, split_y))
        
        # 转换为Tkinter可用的图像格式
        self.app.image_tk[0] = ImageTk.PhotoImage(combined_img)
        
        # 显示图片
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.app.image_tk[0])
        
        # 计算分割线位置
        line_y = (canvas_height - display_height) // 2 + split_y
        
        # 绘制分割线
        canvas.create_line((canvas_width - display_width) // 2, line_y, 
                          (canvas_width + display_width) // 2, line_y, 
                          width=3, fill="white", dash=(5, 5))
    
    def _display_original_only(self, index, canvas, canvas_width, canvas_height, zoom):
        """仅显示原图"""
        original_img = self.app.original_images[index].copy()
        
        # 计算应用缩放后的尺寸
        orig_width, orig_height = original_img.size
        new_width = int(orig_width * zoom)
        new_height = int(orig_height * zoom)
        
        # 检查计算后的尺寸是否有效
        if new_width <= 0 or new_height <= 0:
            raise ValueError("计算后的图片尺寸无效")
        
        # 计算图片在画布上的缩放比例以适应显示
        scale_width = canvas_width / new_width
        scale_height = canvas_height / new_height
        scale = min(scale_width, scale_height, 1.0)  # 不超过原始大小
        
        # 计算最终显示尺寸
        display_width = int(new_width * scale)
        display_height = int(new_height * scale)
        
        # 调整图片大小
        resized_img = original_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter可用的图像格式
        self.app.image_tk[index] = ImageTk.PhotoImage(resized_img)
        
        # 清除画布并显示新图片
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.app.image_tk[index])
    
    def _display_processed_image(self, index, canvas, canvas_width, canvas_height, zoom):
        """显示处理后的图片"""
        # 复制图片以避免修改原始图片
        img = self.app.processed_images[index].copy()
        
        # 计算应用缩放后的尺寸
        orig_width, orig_height = img.size
        new_width = int(orig_width * zoom)
        new_height = int(orig_height * zoom)
        
        # 检查计算后的尺寸是否有效
        if new_width <= 0 or new_height <= 0:
            raise ValueError("计算后的图片尺寸无效")
        
        # 计算图片在画布上的缩放比例以适应显示
        scale_width = canvas_width / new_width
        scale_height = canvas_height / new_height
        scale = min(scale_width, scale_height, 1.0)  # 不超过原始大小
        
        # 计算最终显示尺寸
        display_width = int(new_width * scale)
        display_height = int(new_height * scale)
        
        # 调整图片大小
        resized_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter可用的图像格式
        self.app.image_tk[index] = ImageTk.PhotoImage(resized_img)
        
        # 清除画布并显示新图片
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.app.image_tk[index])
    
    def on_slider_line_click(self, event):
        """分割线点击事件"""
        # 只有在滑动对比模式下才响应
        if self.app.compare_mode.get().startswith("slider_"):
            self.is_dragging_line = True
            self.update_slider_position(event.x, event.y)
    
    def on_slider_line_drag(self, event):
        """分割线拖动事件"""
        if self.is_dragging_line:
            self.update_slider_position(event.x, event.y)
    
    def update_slider_position(self, x, y):
        """更新分割线位置并重新绘制"""
        mode = self.app.compare_mode.get()
        canvas = self.app.image_labels[0]  # 使用第一个画布进行滑动对比
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if mode == "slider_horizontal":
            # 水平滑动对比（左右）
            self.drag_line_pos = max(0.1, min(0.9, x / canvas_width))
        else:
            # 垂直滑动对比（上下）
            self.drag_line_pos = max(0.1, min(0.9, y / canvas_height))
        
        # 重新显示图片以更新滑动效果
        if self.app.original_images[0] is not None:
            self.display_image(0)



    """处理鼠标滚轮缩放---日后添加功能,现在没有用上"""
    def on_mousewheel(self, event, index):
        """处理鼠标滚轮缩放"""
        # 只有当图片被选中时才允许缩放
        if index != self.app.selected_image_index:
            self.app.select_image(index)
        
        # 获取当前缩放值
        current_zoom = self.app.params["zoom"][index]
        
        # 根据滚轮方向调整缩放
        if event.delta > 0:
            # 向上滚动，放大
            new_zoom = min(current_zoom + 0.1, 3.0)
        else:
            # 向下滚动，缩小
            new_zoom = max(current_zoom - 0.1, 0.1)
        
        # 更新缩放参数
        self.app.params["zoom"][index] = new_zoom
        self.app.param_controls["zoom"].set(new_zoom)
        self.app.param_controls["zoom_value"].config(text=f"{new_zoom:.1f}")
        
        # 重新显示图片
        self.display_image(index)
    
    def on_canvas_click(self, event, index):
        """处理画布点击事件"""
        # 记录鼠标位置用于拖动
        self.last_x = event.x
        self.last_y = event.y
        self.is_dragging = False
        
        # 选择图片
        self.app.ui_manager.select_image(index)
    
    def on_drag(self, event, index):
        """处理鼠标拖动"""
        if not self.is_dragging:
            # 检查是否移动了足够距离来触发拖动
            dx = abs(event.x - self.last_x)
            dy = abs(event.y - self.last_y)
            if dx > 3 or dy > 3:  # 小阈值避免点击时触发
                self.is_dragging = True
        
        if self.is_dragging and self.app.params["zoom"][index] > 1.0:
            # 只有当图片被放大时才允许拖动
            canvas = self.app.image_labels[index]
            dx = self.last_x - event.x
            dy = self.last_y - event.y
            
            # 移动画布视图
            canvas.xview_scroll(int(dx / 10), "units")
            canvas.yview_scroll(int(dy / 10), "units")
            
            # 更新鼠标位置
            self.last_x = event.x
            self.last_y = event.y
    
    def on_drag_release(self, event):
        """处理鼠标释放事件"""
        self.is_dragging = False
        self.is_dragging_line = False