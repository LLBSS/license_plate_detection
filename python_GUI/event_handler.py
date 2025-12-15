import tkinter as tk
from tkinter import filedialog

class EventHandler:
    """事件处理器类，负责处理应用中的各种事件"""
    
    def __init__(self, app):
        self.app = app
    
    def setup_event_bindings(self):
        """设置所有事件绑定"""
        # 设置键盘事件
        self._setup_keyboard_events()
        
        # 设置窗口事件
        self._setup_window_events()
        
        # 设置拖放功能
        self._setup_drag_and_drop()
    
    def _setup_keyboard_events(self):
        """设置键盘事件处理"""
        # 绑定数字键1和2用于选择图片
        self.app.root.bind("1", lambda e: self._select_image(0))
        self.app.root.bind("2", lambda e: self._select_image(1))
        
        # 绑定R键重置参数
        self.app.root.bind("r", lambda e: self.app.reset_params())
        
        # 绑定Ctrl+O快捷键打开图片
        self.app.root.bind("<Control-o>", lambda e: self.app.open_image(self.app.selected_image_index))
        
        # 绑定Ctrl+S快捷键保存图片
        self.app.root.bind("<Control-s>", lambda e: self.app.save_image())
        
        # 绑定方向键微调参数
        # 全部未实现
        self.app.root.bind("<Up>", lambda e: self._adjust_current_param(0.05))
        self.app.root.bind("<Down>", lambda e: self._adjust_current_param(-0.05))
        self.app.root.bind("<Left>", lambda e: self._adjust_current_param(-0.01))
        self.app.root.bind("<Right>", lambda e: self._adjust_current_param(0.01))
    
    #已使用--设置窗口事件处理
    def _setup_window_events(self):
        """设置窗口事件处理"""
        # 窗口大小调整事件
        self.app.root.bind("<Configure>", self.app.on_window_resize)
        
        # 窗口关闭事件
        self.app.root.protocol("WM_DELETE_WINDOW", self.app.on_window_close)
    
    #未使用--设置拖放功能
    def _setup_drag_and_drop(self):
        """设置拖放功能
        注意：完整的拖放功能在标准Tkinter中有限制，这里暂不实现完整拖放"""
        # 拖放功能需要额外的TkinterDND库或其他方式实现
        # 暂时不绑定不支持的事件
    
    def _select_image(self, index):
        """选择指定索引的图片"""
        if index in [0, 1] and self.app.original_images[index] is not None:
            self.app.selected_image_index = index
            self.app.ui_manager.select_image(index)
            self.app.ui_manager.update_status(f"已选择图片 {index + 1}")
    
    #已使用--微调当前选中的参数
    def _adjust_current_param(self, delta):
        """微调当前选中的参数"""
        # 获取当前选中的参数控件（如果有）
        # 这里简化处理，假设当前调整的是缩放参数
        current_param = "zoom"
        self.app.adjust_param(current_param, delta)
    
    #未使用--拖入事件处理
    def _on_drag_enter(self, event):
        """拖入事件处理"""
        # 接受文件拖放
        event.widget.focus_force()
        event.accept()
    
    #未使用--拖离事件处理
    def _on_drag_leave(self, event):
        """拖离事件处理"""
        pass
    
    #未使用--拖放完成事件处理
    def _on_drop(self, event):
        """拖放完成事件处理"""
        try:
            # 获取拖放的文件路径
            file_path = event.data
            
            # Windows系统中路径可能以"{...}"格式传递，需要处理
            if file_path.startswith("{") and file_path.endswith("}"):
                file_path = file_path[1:-1]
            
            # 检查文件格式
            if self._is_supported_image(file_path):
                # 加载图片到当前选中的位置
                self.app._load_image(file_path, self.app.selected_image_index)
                self.app.ui_manager.update_status(f"已加载图片: {file_path}")
            else:
                self.app.ui_manager.show_error("不支持的图片格式")
        except Exception as e:
            self.app.ui_manager.show_error(f"加载图片失败: {str(e)}")
    
    def _is_supported_image(self, file_path):
        """检查文件是否为支持的图片格式"""
        supported_formats = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        file_extension = file_path.lower().split(".")[-1]
        return any(file_path.lower().endswith(fmt) for fmt in supported_formats)
    
    #未使用--画布点击
    def on_canvas_click(self, event, index):
        """画布点击事件处理"""
        if self.app.original_images[index] is not None:
            # 选中当前图片
            if self.app.selected_image_index != index:
                self.app.selected_image_index = index
                self.app.ui_manager.select_image(index)
    
    #未使用--鼠标滚轮
    def on_mousewheel(self, event, index):   
        """鼠标滚轮事件处理"""
        if self.app.original_images[index] is not None:
            # 获取滚轮方向
            if event.delta > 0:
                zoom_factor = 1.1
            else:
                zoom_factor = 0.9
            
            # 计算新的缩放值
            current_zoom = self.app.params["zoom"][index]
            new_zoom = max(0.1, min(5.0, current_zoom * zoom_factor))
            
            # 更新缩放参数
            self.app.params["zoom"][index] = new_zoom
            
            # 更新UI显示
            self.app.ui_manager.update_param_controls()
            
            # 更新图片显示
            self.app.display_manager.display_image(index)
            
            # 更新状态栏
            self.app.ui_manager.update_status(f"图片 {index + 1} 缩放: {new_zoom:.2f}", timeout=1500)