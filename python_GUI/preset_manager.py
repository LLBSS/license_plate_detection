import os
import json
import tkinter as tk
from tkinter import simpledialog, ttk

class PresetManager:
    """预设管理类，负责保存和加载用户的图片处理参数预设"""
    
    def __init__(self, app):
        self.app = app
        self.preset_dir = os.path.join(os.path.expanduser("~"), ".image_editor_presets")
    
    def save_preset(self):
        """保存当前参数预设"""
        try:
            idx = self.app.selected_image_index
            
            if self.app.original_images[idx] is None:
                self.app.show_info("提示", "请先选择一张图片")
                return
            
            # 获取预设名称
            preset_name = simpledialog.askstring("保存预设", "请输入预设名称:")
            
            if preset_name:
                try:
                    # 创建预设数据（包含滤镜信息）
                    preset_data = {
                        "name": preset_name,
                        "params": {
                            "brightness": self.app.params["brightness"][idx],
                            "contrast": self.app.params["contrast"][idx],
                            "saturation": self.app.params["saturation"][idx],
                            "sharpness": self.app.params["sharpness"][idx],
                            "filter": self.app.params["filter"][idx]  # 添加滤镜信息
                        }
                    }
                    
                    # 保存预设到文件
                    os.makedirs(self.preset_dir, exist_ok=True)
                    preset_file = os.path.join(self.preset_dir, f"{preset_name.replace(' ', '_')}.json")
                    
                    with open(preset_file, 'w', encoding='utf-8') as f:
                        json.dump(preset_data, f, ensure_ascii=False, indent=2)
                    
                    self.app.show_info("成功", f"预设 '{preset_name}' 已保存")
                    
                except Exception as e:
                    self.app.show_error("保存失败", f"保存预设时出错:\n{str(e)}")
        except Exception as e:
            self.app.show_error("操作失败", f"保存预设时出错:\n{str(e)}")
    
    def load_preset(self):
        """加载参数预设"""
        try:
            idx = self.app.selected_image_index
            
            if self.app.original_images[idx] is None:
                self.app.show_info("提示", "请先选择一张图片")
                return
            
            # 检查预设目录是否存在
            if not os.path.exists(self.preset_dir):
                self.app.show_info("提示", "没有找到保存的预设")
                return
            
            # 获取所有预设文件
            presets = []
            for file in os.listdir(self.preset_dir):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(self.preset_dir, file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            presets.append(data['name'])
                    except:
                        continue
            
            if not presets:
                self.app.show_info("提示", "没有找到有效的预设")
                return
            
            # 显示预设选择对话框
            dialog = PresetDialog(self.app.root, "加载预设", presets)
            
            if dialog.selected_preset:
                try:
                    # 读取预设文件
                    preset_file = os.path.join(self.preset_dir, f"{dialog.selected_preset.replace(' ', '_')}.json")
                    
                    with open(preset_file, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)
                    
                    # 应用预设参数
                    params = preset_data.get('params', {})
                    for param, value in params.items():
                        if param in self.app.params:
                            self.app.params[param][idx] = value
                            # 如果是滤镜参数，更新滤镜选择框
                            if param == "filter" and hasattr(self.app, 'filter_var'):
                                self.app.filter_var.set(value)
                    
                    # 更新UI
                    self.app.update_param_controls()
                    self.app.process_image(idx)
                    
                    self.app.show_info("成功", f"已应用预设 '{dialog.selected_preset}'")
                    
                except Exception as e:
                    self.app.show_error("加载失败", f"加载预设时出错:\n{str(e)}")
        except Exception as e:
            self.app.show_error("操作失败", f"加载预设时出错:\n{str(e)}")


class PresetDialog(simpledialog.Dialog):
    """预设选择对话框"""
    def __init__(self, parent, title=None, presets=None):
        self.presets = presets
        self.selected_preset = None
        super().__init__(parent, title)
    
    def body(self, master):
        ttk.Label(master, text="选择预设:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.preset_list = tk.Listbox(master, width=30, height=10)
        self.preset_list.grid(row=1, column=0, sticky=tk.NSEW, pady=5)
        
        scrollbar = ttk.Scrollbar(master, orient=tk.VERTICAL, command=self.preset_list.yview)
        scrollbar.grid(row=1, column=1, sticky=tk.NS)
        self.preset_list.config(yscrollcommand=scrollbar.set)
        
        # 填充预设列表
        for preset in self.presets:
            self.preset_list.insert(tk.END, preset)
        
        # 默认选中第一个
        if self.presets:
            self.preset_list.selection_set(0)
        
        return self.preset_list
    
    def apply(self):
        selection = self.preset_list.curselection()
        if selection:
            self.selected_preset = self.presets[selection[0]]