import tkinter as tk
from tkinter import ttk, messagebox

class UIManager:
    """UI管理类，负责创建和管理应用的用户界面组件"""
    
    def __init__(self, app):
        self.app = app
        self.status_text = tk.StringVar()
        self.filter_var = tk.StringVar()
        # 引用app上的image_titles属性
        self.image_titles = []
    
    def create_main_layout(self):
        """创建主布局"""
        # 创建主框架
        main_frame = ttk.Frame(self.app.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建顶部控制区
        self._create_top_control_area(main_frame)
        
        # 创建图片显示区域
        self._create_image_display_area(main_frame)
        
        # 创建参数调节控制面板
        self._create_control_panel(main_frame)
        
        # 创建状态栏
        self._create_status_bar()
        
        # 设置初始状态文本
        self.status_text.set("就绪 - 请选择图片开始编辑")

    #<-               副函数1111               ->
    def _create_top_control_area(self, parent):
        """创建顶部控制区域"""
        top_frame = ttk.Frame(parent, padding="5")
        top_frame.pack(fill=tk.X, pady=5)
        
        # 创建图片选择按钮
        self.app.btn_open_image1 = ttk.Button(top_frame, text="打开图片1", command=lambda: self.app.open_image(0))
        self.app.btn_open_image1.pack(side=tk.LEFT, padx=5)
        
        self.app.btn_open_image2 = ttk.Button(top_frame, text="打开图片2", command=lambda: self.app.open_image(1))
        self.app.btn_open_image2.pack(side=tk.LEFT, padx=5)
        
        # 对比模式选择
        self._create_compare_mode_selection(top_frame)
        
        # 帮助按钮
        ttk.Button(top_frame, text="帮助", command=self.app.show_help).pack(side=tk.RIGHT, padx=5)
    
    def _create_compare_mode_selection(self, parent):
        """创建对比模式选择控件"""
        compare_frame = ttk.LabelFrame(parent, text="对比模式")
        compare_frame.pack(side=tk.LEFT, padx=10)
        
        self.app.compare_mode = tk.StringVar(value="side_by_side")
        ttk.Radiobutton(compare_frame, text="并排对比", variable=self.app.compare_mode, value="side_by_side", 
                       command=self.app.on_compare_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(compare_frame, text="左右滑动对比", variable=self.app.compare_mode, value="slider_horizontal", 
                       command=self.app.on_compare_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(compare_frame, text="上下滑动对比", variable=self.app.compare_mode, value="slider_vertical", 
                       command=self.app.on_compare_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(compare_frame, text="原图", variable=self.app.compare_mode, value="original", 
                       command=self.app.on_compare_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(compare_frame, text="处理图", variable=self.app.compare_mode, value="processed", 
                       command=self.app.on_compare_mode_change).pack(side=tk.LEFT, padx=5)
    
    def _create_image_display_area(self, parent):
        """创建图片显示区域"""
        self.app.image_frame = ttk.Frame(parent)
        self.app.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建两个图片容器
        self.app.image_containers = [ttk.Frame(self.app.image_frame) for _ in range(2)]
        self.app.image_titles = [None, None]  # 存储图片标题
        self.app.image_labels = [None, None]  # 存储图片标签
        # 更新UIManager实例上的image_titles引用
        self.image_titles = self.app.image_titles
        
        for i, container in enumerate(self.app.image_containers):
            container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # 图片标题
            title_frame = ttk.Frame(container)
            title_frame.pack(fill=tk.X, pady=5)
            self.app.image_titles[i] = ttk.Label(title_frame, text=f"图片 {i+1}")
            self.app.image_titles[i].pack(anchor=tk.W)
            
            # 图片显示标签
            image_scroll_frame = ttk.Frame(container)
            image_scroll_frame.pack(fill=tk.BOTH, expand=True)
            
            # 添加滚动条
            h_scroll = ttk.Scrollbar(image_scroll_frame, orient=tk.HORIZONTAL)
            v_scroll = ttk.Scrollbar(image_scroll_frame, orient=tk.VERTICAL)
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 创建画布用于显示图片
            self.app.image_labels[i] = tk.Canvas(image_scroll_frame, bg="#f0f0f0", 
                                           xscrollcommand=h_scroll.set, 
                                           yscrollcommand=v_scroll.set)
            self.app.image_labels[i].pack(fill=tk.BOTH, expand=True)
            
            # 配置滚动条
            h_scroll.config(command=self.app.image_labels[i].xview)
            v_scroll.config(command=self.app.image_labels[i].yview)
            
            # 添加事件绑定
            self.app.image_labels[i].bind("<Button-1>", lambda e, idx=i: self.app.display_manager.on_canvas_click(e, idx))
            self.app.image_labels[i].bind("<Double-1>", lambda e, idx=i: self.app.open_image(idx))
            self.app.image_labels[i].bind("<MouseWheel>", lambda e, idx=i: self.app.display_manager.on_mousewheel(e, idx))
            self.app.image_labels[i].bind("<B1-Motion>", lambda e, idx=i: self.app.display_manager.on_drag(e, idx))
            self.app.image_labels[i].bind("<ButtonRelease-1>", self.app.display_manager.on_drag_release)
    
    def _create_control_panel(self, parent):
        """创建参数调节控制面板"""
        control_frame = ttk.LabelFrame(parent, text="参数调节", padding="10")
        control_frame.pack(fill=tk.X, pady=10)
        
        # 参数控制区域
        self.app.param_controls = {}
        self._create_param_controls(control_frame)
        
        # 底部操作按钮
        self._create_bottom_buttons(control_frame)
    

    #<-               副函数参数创建               ->
    def _create_param_controls(self, parent):
        """创建参数控制组件"""
        param_names = {
            "brightness": "亮度",
            "contrast": "对比度",
            "saturation": "饱和度",
            "sharpness": "清晰度",
            "zoom": "缩放比例"
        }
        
        # 先定义参数框架
        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.X)
        # 滤镜选择
        self._create_filter_selection(param_frame)
        # 创建每行两个参数控制，从第1行开始，避免覆盖滤镜选择
        for i, (param_key, param_label) in enumerate(param_names.items()):
            row = (i // 2) + 1  # 从第1行开始
            col = i % 2
            
            param_item_frame = ttk.Frame(param_frame)
            param_item_frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            # 参数标签
            ttk.Label(param_item_frame, text=param_label).pack(anchor=tk.W)
            
            # 参数控制行
            control_row = ttk.Frame(param_item_frame)
            control_row.pack(fill=tk.X, pady=2)
            
            # 减少按钮
            ttk.Button(control_row, text="-", width=3, 
                      command=lambda p=param_key: self.app.adjust_param(p, -0.1)).pack(side=tk.LEFT)
            
            # 滑块
            self.app.param_controls[param_key] = tk.DoubleVar(value=1.0)
            scale = ttk.Scale(control_row, from_=0.1, to=3.0, orient=tk.HORIZONTAL,
                            variable=self.app.param_controls[param_key], length=200,
                            command=lambda val, p=param_key: self.app.on_param_change(p, val))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # 数值显示
            self.app.param_controls[f"{param_key}_value"] = ttk.Label(control_row, text="1.0", width=5)
            self.app.param_controls[f"{param_key}_value"].pack(side=tk.LEFT, padx=5)
            
            # 增加按钮
            ttk.Button(control_row, text="+", width=3,
                      command=lambda p=param_key: self.app.adjust_param(p, 0.1)).pack(side=tk.LEFT)
        
        # 设置权重使网格扩展均匀
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=1)
    
    def _create_filter_selection(self, parent):
        """创建车牌检测结果选择组件"""
        filter_frame = ttk.Frame(parent)
        filter_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew", columnspan=2)
        ttk.Label(filter_frame, text="检测结果类型:").pack(side=tk.LEFT, padx=5)
        # 车牌检测结果类型
        result_types = [
            "gray_image", "white_mask", "canny_edges", "opened_edges",
            "closed_edges", "vertical_edges", "enhanced_vertical", "contours","combined_result",
            "second_closed_edge"
        ]
        self.app.filter_var = tk.StringVar(value="gray_image")
        filter_combobox = ttk.Combobox(filter_frame, textvariable=self.app.filter_var, values=result_types, state="readonly", width=20)
        filter_combobox.pack(side=tk.LEFT, padx=5)
        filter_combobox.bind("<<ComboboxSelected>>", self.app.on_filter_change)
    
        #<-               副函数底部按钮创建               ->
    def _create_bottom_buttons(self, parent):
        """创建底部操作按钮"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        # 重置按钮
        ttk.Button(bottom_frame, text="重置参数", command=self.app.reset_params).pack(side=tk.LEFT, padx=5)
        
        # 保存按钮
        ttk.Button(bottom_frame, text="保存图片", command=self.app.save_image).pack(side=tk.LEFT, padx=5)
        
        # 预设按钮
        ttk.Button(bottom_frame, text="保存预设", command=self.app.preset_manager.save_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="加载预设", command=self.app.preset_manager.load_preset).pack(side=tk.LEFT, padx=5)
        
        # 清空图片按钮
        ttk.Button(bottom_frame, text="清空当前图片", command=self.app.clear_current_image).pack(side=tk.LEFT, padx=5)
        # 车牌图片处理按钮
        ttk.Button(bottom_frame, text="处理车牌图片", command=self.app.process_license_plate).pack(side=tk.LEFT, padx=5)    
        

    #<-               副函数状态栏创建               ->
    def _create_status_bar(self):
        """创建状态栏"""
        status_bar = ttk.Frame(self.app.root, relief=tk.SUNKEN, height=25)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status_bar, textvariable=self.status_text, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=2)
    
    def update_status(self, message, timeout=3000):
        """更新状态栏消息，可选择自动消失"""
        self.status_text.set(message)
        if timeout > 0:
            self.app.root.after(timeout, lambda: self.status_text.set("就绪 - 请选择图片开始编辑"))
    
    def select_image(self, index):
        """更新UI以显示当前选中的图片"""
        for i in range(2):
            if i == index:
                self.app.image_containers[i].config(style="Selected.TFrame")
                self.app.image_titles[i].config(font=("SimHei", 10, "bold"))
            else:
                self.app.image_containers[i].config(style="TFrame")
                self.app.image_titles[i].config(font=("SimHei", 10))
        
        # 更新参数控制器的值
        self.update_param_controls()
    
    def update_param_controls(self):
        """更新参数控制器显示当前选中图片的参数"""
        idx = self.app.selected_image_index
        for param in ["brightness", "contrast", "saturation", "sharpness", "zoom"]:
            self.app.param_controls[param].set(self.app.params[param][idx])
            self.app.param_controls[f"{param}_value"].config(text=f"{self.app.params[param][idx]:.1f}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
图片编辑与对比应用帮助：

基本操作：
  • 点击"打开图片1"或"打开图片2"选择图片
  • 使用键盘1和2键快速切换选中的图片
  • 拖动滑块或点击+/-按钮调整图片参数
  • 点击"重置参数"恢复默认设置
  • 点击"保存图片"保存编辑后的图片

快捷操作：
  • Ctrl+O: 打开图片
  • Ctrl+S: 保存图片
  • R: 重置参数
  • 鼠标滚轮: 缩放图片
  • 拖动鼠标: 移动放大后的图片
  • 拖放文件: 直接加载图片

对比模式：
  • 并排对比: 同时显示两张图片
  • 原图/处理图: 显示原图和处理后的对比

支持格式：JPG, PNG, GIF, BMP等
        """
        messagebox.showinfo("使用帮助", help_text)
    
    def show_error(self, message):
        """显示错误消息"""
        messagebox.showerror("错误", message)