# coding=utf-8
import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog
#计算阈值
def calculate_brightness_threshold(image_path):
    """
    计算图片的亮度阈值
    :param image_path: 图片路径
    :return: 包含阈值信息的字典
    """
    if not os.path.exists(image_path):
        print(f"错误：图片路径 {image_path} 不存在")
        return None
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    
    # 将图片转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算亮度直方图的90%分位数作为阈值
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_cumulative = np.cumsum(hist)
    total_pixels = gray_image.size
    threshold_90 = np.where(hist_cumulative >= total_pixels * 0.9)[0][0]
    
    # 计算平均亮度和标准差
    mean_brightness = np.mean(gray_image)
    std_brightness = np.std(gray_image)
    
    # 根据平均亮度和标准差计算阈值
    adaptive_threshold = int(mean_brightness + std_brightness)
    adaptive_threshold = max(0, min(255, adaptive_threshold))  # 确保在0-255范围内
    
    # 返回结果
    result = {
        'filename': os.path.basename(image_path),
        'threshold_90': int(threshold_90),
        'adaptive_threshold': adaptive_threshold
    }
    
    return result

def process_single_image():
    """
    处理单张图片
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        title="选择图片"
    )
    
    if not file_path:
        print("未选择图片，程序退出")
        return
    
    # 计算阈值
    result = calculate_brightness_threshold(file_path)
    
    if result:
        print(f"图片: {result['filename']}")
        print(f"90%分位数阈值: {result['threshold_90']}")
        print(f"自适应阈值(均值+标准差): {result['adaptive_threshold']}")
        
        # 保存结果到文件
        with open("brightness_threshold_result.txt", "w") as f:
            f.write(f"图片路径: {file_path}\n")
            f.write(f"90%分位数阈值: {result['threshold_90']}\n")
            f.write(f"自适应阈值(均值+标准差): {result['adaptive_threshold']}\n")
        print(f"阈值结果已保存到 brightness_threshold_result.txt")

def batch_process_images(folder_path):
    """
    批量处理文件夹中的图片
    :param folder_path: 包含图片的文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"错误：文件夹路径 {folder_path} 不存在")
        return
    
    # 获取所有图片文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"错误：在文件夹 {folder_path} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始批量处理...")
    
    # 保存所有结果
    all_results = []
    
    for image_file in image_files:
        result = calculate_brightness_threshold(image_file)
        if result:
            all_results.append(result)
            print(f"图片: {result['filename']} -> 90%阈值: {result['threshold_90']}, 自适应阈值: {result['adaptive_threshold']}")
    
    # 将结果保存到文件
    output_file = "brightness_threshold_batch_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("批量亮度阈值计算结果\n")
        f.write("=" * 50 + "\n")
        f.write("文件名\t90%分位数阈值\t自适应阈值(均值+标准差)\n")
        
        for result in all_results:
            f.write(f"{result['filename']}\t{result['threshold_90']}\t{result['adaptive_threshold']}\n")
    
    print(f"\n批量处理完成！结果已保存到 {output_file}")
    return all_results

def main():
    """
    主函数
    """
    print("亮度阈值计算器")
    print("=" * 30)
    
    # 检查是否有examples文件夹
    examples_folder = "examples"
    if os.path.exists(examples_folder):
        print(f"发现examples文件夹，将进行批量处理")
        batch_process_images(examples_folder)
    else:
        # 检查是否有examples-canny文件夹
        examples_canny_folder = "examples-canny"
        if os.path.exists(examples_canny_folder):
            print(f"未找到examples文件夹，发现examples-canny文件夹，将进行批量处理")
            batch_process_images(examples_canny_folder)
        else:
            # 检查是否有test文件夹
            test_folder = "test"
            if os.path.exists(test_folder):
                print(f"未找到examples和examples-canny文件夹，发现test文件夹，将进行批量处理")
                batch_process_images(test_folder)
            else:
                print("未找到examples、examples-canny或test文件夹，将处理单张图片")
                process_single_image()

if __name__ == "__main__":
    main()