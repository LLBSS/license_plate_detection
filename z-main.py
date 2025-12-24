# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# 导入自定义函数
from z_preprocess_plate import preprocess_plate
from z_detect_plate_region import detect_plate_region
from z_segment_chars import segment_chars
from z_recognize_chars import recognize_chars

def main():
    """
    主函数，对应MATLAB的main.m
    功能：创建车牌识别系统的UI界面和处理流程
    """
    # ================= 创建 UI 界面框架 =================
    root = tk.Tk()
    root.title("车牌识别系统 —— 成都理工大学")
    root.geometry("1000x700")
    root.configure(bg="#F0F0F0")
    
    # 1. 顶部按钮
    btn_select = tk.Button(root, text="选择图片并识别", font=("Arial", 12, "bold"), 
                          command=lambda: btn_callback(root, h_text_result, ax_orig, ax_edge, ax_morph, 
                                                      ax_plate_gray, ax_plate_bw, ax_chars))
    btn_select.place(x=20, y=20, width=150, height=50)
    
    # 2. 结果显示文本框
    h_text_result = tk.Label(root, text="等待选择图片...", font=("Arial", 18, "bold"), 
                           fg="blue", bg="#F0F0F0", anchor="w")
    h_text_result.place(x=200, y=20, width=700, height=60)
    
    # ================= 3. 创建显示图像的画布 =================
    
    # --- 第一行：预处理全貌 ---
    # 1.1 原始图像
    ax_orig = create_image_frame(root, "1. 原始图像", 20, 100, 300, 280)
    
    # 1.2 边缘检测
    ax_edge = create_image_frame(root, "2. 边缘检测(Sobel)", 350, 100, 300, 280)
    
    # 1.3 形态学滤波结果
    ax_morph = create_image_frame(root, "3. 形态学滤波(定位区域)", 680, 100, 300, 280)
    
    # --- 第二行：车牌提取详情 ---
    # 2.1 切割出的灰度车牌
    ax_plate_gray = create_image_frame(root, "4. 车牌截取(灰度)", 150, 400, 300, 200)
    
    # 2.2 二值化后的车牌
    ax_plate_bw = create_image_frame(root, "5. 车牌二值化与去边框", 550, 400, 300, 200)
    
    # --- 第三行：字符分割 ---
    ax_chars = []
    char_width = 80
    start_x = 220
    for i in range(7):
        ax = create_image_frame(root, f"字符{i+1}", start_x + i*char_width, 620, 70, 150)
        ax_chars.append(ax)
    
    # 启动主循环
    root.mainloop()

def create_image_frame(parent, title, x, y, width, height):
    """
    创建带标题的图像显示框
    参数：parent - 父窗口，title - 标题，x,y - 位置，width,height - 尺寸
    返回：画布对象
    """
    # 创建框架
    frame = tk.Frame(parent, bg="white", bd=2, relief="groove")
    frame.place(x=x, y=y, width=width, height=height)
    
    # 添加标题
    title_label = tk.Label(frame, text=title, font=("Arial", 10, "bold"), bg="white")
    title_label.pack(side="top", fill="x")
    
    # 创建画布
    canvas = tk.Canvas(frame, bg="white", bd=0)
    canvas.pack(side="bottom", fill="both", expand=True)
    
    return canvas

def display_image(canvas, image):
    """
    在画布上显示图像
    参数：canvas - 画布对象，image - OpenCV图像
    """
    # 清除画布
    canvas.delete("all")
    
    if image is None:
        return
    
    # 转换图像格式
    if len(image.shape) == 3:
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        # 灰度图转RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(image)
    
    # 调整图像大小以适应画布
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    if canvas_width > 1 and canvas_height > 1:
        pil_image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
    
    # 转换为Tkinter图像
    tk_image = ImageTk.PhotoImage(pil_image)
    
    # 保存图像引用（防止被垃圾回收）
    canvas.image = tk_image
    
    # 显示图像
    canvas.create_image(canvas_width//2, canvas_height//2, image=tk_image, anchor="center")

def btn_callback(root, h_text_result, ax_orig, ax_edge, ax_morph, 
                 ax_plate_gray, ax_plate_bw, ax_chars):
    """
    按钮回调函数，对应MATLAB的btnCallback
    功能：处理图片选择和车牌识别流程
    """
    # 选择图片文件
    filename = filedialog.askopenfilename(
        filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")],
        title="选择车牌图像"
    )
    
    if not filename:
        return
    
    try:
        h_text_result.config(text="正在处理...", fg="black")
        root.update()
        
        # 读取图像
        I = cv2.imread(filename)
        if I is None:
            messagebox.showerror("错误", "无法读取选择的图像文件")
            return
        
        # 显示原始图像
        display_image(ax_orig, I)
        
        # 1. 预处理 (获取 I_final 和 I_gray)
        I_final, I_gray = preprocess_plate(I)
        
        # 边缘检测结果（使用预处理后的I_gray进行边缘检测）
        I_edge_display = cv2.Canny(I_gray, 50, 150)
        
        # 显示预处理结果
        display_image(ax_edge, I_edge_display)
        display_image(ax_morph, I_final)
        
        # 2. 车牌定位 (获取 二值化车牌 和 灰度车牌)
        I_plate_bw, I_plate_gray_out = detect_plate_region(I, I_final)
        
        if I_plate_bw is None:
            h_text_result.config(text="错误：未能检测到车牌区域", fg="red")
            return
        
        # 显示车牌定位结果
        display_image(ax_plate_gray, I_plate_gray_out)
        display_image(ax_plate_bw, I_plate_bw)
        
        # 3. 字符分割
        char_imgs = segment_chars(I_plate_bw)
        
        # 显示分割后的字符
        for i in range(7):
            if i < len(char_imgs) and char_imgs[i] is not None:
                display_image(ax_chars[i], char_imgs[i])
            else:
                # 清空画布
                ax_chars[i].delete("all")
        
        # 4. 识别
        char_result = recognize_chars(char_imgs)
        h_text_result.config(text=f"识别结果: {char_result}", fg="red")
        print(f"识别结果: {char_result}")
        
    except Exception as e:
        messagebox.showerror("错误", str(e))
        print(f"错误: {e}")

if __name__ == "__main__":
    main()