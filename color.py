import argparse
import numpy as np
import cv2
from color_detector import WhitePlateDetector
import os

def process_image(image_path):
    image_original = cv2.imread(image_path)
    if image_original is None:
        print(f"无法加载图像: {image_path}")
        return None
    detector = WhitePlateDetector()
    image = detector.detect_plates_by_white_colors(image_original)
    return image

def main():
    path = './examples'  # 替换为你的图像路径

    if not os.path.exists('E:\A-E file\Trae CN Project\car_detection\examples-color'):
        os.makedirs('E:\A-E file\Trae CN Project\car_detection\examples-color')

    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(path, filename)
            image = process_image(image_path)
            if image is not None:
                # 保存处理后的图像
                image_save_path = 'E:\A-E file\Trae CN Project\car_detection\examples-color\\'+filename
                cv2.imwrite(image_save_path, image['white'])
                print(f"处理后的图像已保存至: {image_save_path}")
# 使用示例
if __name__ == "__main__":
    main()

