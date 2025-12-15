import cv2
import numpy as np
import argparse
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from detectors import LicensePlateDetector
from image_utils import show_images
from color_detector import WhitePlateDetector
def main():
    parser = argparse.ArgumentParser(description='车牌定位检测系统')
    parser.add_argument('--image_path', default='examples/car6.jpg', help='输入图像路径')
    parser.add_argument('--low-threshold', type=int, default=50, help='Canny低阈值')
    parser.add_argument('--high-threshold', type=int, default=150, help='Canny高阈值')
    parser.add_argument('--blur-size', type=int, default=5, help='高斯模糊内核大小')
    parser.add_argument('--simple', action='store_true', help='简化模式：只显示最终结果，不使用网格显示')
    parser.add_argument('--no-display', action='store_true', help='不显示任何窗口')
    parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    
    args = parser.parse_args()
    
    try:
        # 加载图像
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"图像文件不存在: {args.image_path}")
        
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {args.image_path}")
        
        print(f"✅ 图像加载成功: {args.image_path}")
        print(f"📐 图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 创建车牌检测器
        detector = LicensePlateDetector(
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            blur_size=args.blur_size
        )
        
        # 执行检测
        print("🔍 正在检测车牌...")
        results = detector.detect_license_plates(image)
        print(results.keys())
        #print(results)
        # 显示所有处理步骤
        show_images(results, "车牌检测全过程")
        # 保存结果
        # if not args.no_save:
        #     save_results = detector.save_detection_results(image, results, args.image_path)
        #     print(f"💾 结果已保存到: {save_results['output_image']}")
        #     print(f"💾 坐标数据已保存到: {save_results['output_json']}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()