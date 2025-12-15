import cv2
import numpy as np
import os
import json
from datetime import datetime
import image_utils
import color_detector
import argparse


class LicensePlateDetector:
    def __init__(self, low_threshold=50, high_threshold=150, blur_size=5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.blur_size = blur_size
    
    def preprocess_image(self, image):
        """图像预处理：灰度化 + 高斯模糊"""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # 高斯模糊降噪
        blurred_image = cv2.GaussianBlur(gray_image, (self.blur_size, self.blur_size), 0)
        return gray_image, blurred_image
    
    def canny_edge_detection(self, image):
        """Canny边缘检测"""
        gray_image, blurred_image = self.preprocess_image(image)
        edges = cv2.Canny(blurred_image, self.low_threshold, self.high_threshold, apertureSize=3)
        return gray_image, blurred_image, edges
    
    def morphological_operations(self, edges):
        """
        执行开运算和闭运算的形态学操作
        开运算：先腐蚀后膨胀，用于去除噪声、孤立点
        闭运算：先膨胀后腐蚀，用于连接小的间隙
        """
        # 开运算：去除小噪声，保留大的边缘结构
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, open_kernel)
        
        # 闭运算：连接车牌字符之间的间隙，强化车牌区域
        # 使用水平方向的长方形核，更好地连接车牌字符
        
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed_edges = cv2.morphologyEx(opened_edges, cv2.MORPH_CLOSE, close_kernel)
        
        return opened_edges, closed_edges
    
    def detect_vertical_edges(self, gray_image):
        """使用Sobel算子检测垂直边缘，车牌字符主要包含垂直边缘"""
        # Sobel垂直方向算子
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 取绝对值并转换为8位
        vertical_edges = np.absolute(sobel_y)
        vertical_edges = np.uint8(255 * vertical_edges / np.max(vertical_edges))
        
        return vertical_edges
    
    def enhance_license_plate_features(self, vertical_edges):
        """增强车牌特征：强化垂直边缘，适合车牌宽高比"""
        # 二值化突出强垂直边缘
        _, vertical_binary = cv2.threshold(vertical_edges, 80, 255, cv2.THRESH_BINARY)
        
        # 形态学操作增强垂直线条（适合车牌字符）
        # 垂直方向的小内核，强化字符垂直边缘
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        enhanced_vertical = cv2.morphologyEx(vertical_binary, cv2.MORPH_DILATE, vertical_kernel)
        
        # 使用适合车牌形状的闭运算核（长矩形）
        license_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        enhanced_vertical = cv2.morphologyEx(enhanced_vertical, cv2.MORPH_CLOSE, license_kernel)
        
        return enhanced_vertical
    
    def find_contours(self, processed_image):
        """寻找图像中的轮廓"""
        # 寻找所有轮廓
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def filter_license_plate_contours(self, contours, min_area=2000, max_area=50000):
        """
        根据车牌特征过滤轮廓
        车牌特征：矩形形状、特定的宽高比（通常在2.5:1到4:1之间）
        """
        license_plates = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < min_area or area > max_area:
                continue
            
            # 获取最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比
            if h == 0:
                continue
            aspect_ratio = w / h
            
            # 车牌宽高比通常在2.5:1到4:1之间
            # 中国车牌比例约为2.89:1 (440mm×140mm)
            if 2.0 < aspect_ratio < 5.0:
                # 计算轮廓的矩形度（面积与最小外接矩形面积的比值）
                rect_area = w * h
                rect_ratio = area / rect_area if rect_area > 0 else 0
                
                # 车牌应该接近矩形
                if rect_ratio > 0.5:
                    license_plates.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'rect_ratio': rect_ratio,
                        'confidence': rect_ratio * (1 - abs(aspect_ratio - 3.0) / 3.0)  # 基于矩形度和宽高比的置信度
                    })
        
        # 按置信度排序
        license_plates.sort(key=lambda lp: lp['confidence'], reverse=True)
        return license_plates
    
    def detect_license_plates(self, image):
        #print("=== 标记点1: 开始车牌检测 ===")
        """完整的车牌检测流程"""
        # 1. Canny边缘检测
        gray_image, blurred_image, canny_edges = self.canny_edge_detection(image)
        # Step 2: 车牌颜色检测
        detector = color_detector.WhitePlateDetector()
        mask_image = detector.detect_plates_by_white_colors(image)
        white_mask = mask_image['white']
        
        # 2. 形态学操作：开运算和闭运算
        opened_edges, closed_edges = self.morphological_operations(white_mask)
        
        # 3. 垂直边缘检测和增强
        vertical_edges = self.detect_vertical_edges(gray_image)
        enhanced_vertical = self.enhance_license_plate_features(vertical_edges)
        
        # 4. 结合闭运算结果和增强的垂直边缘
        #combined_result = cv2.bitwise_and(closed_edges, enhanced_vertical)
        combined_result = cv2.bitwise_and(opened_edges, enhanced_vertical)

        # 5. 寻找轮廓
        contours = self.find_contours(combined_result)
        first_contour = np.array(contours[0])
        # 6. 过滤出可能的车牌区域
        #license_plates = self.filter_license_plate_contours(contours)
        license_plates = self.filter_license_plate_contours(first_contour)
        license_plates = np.array(license_plates)
        # 7. 二次开运算和闭运算
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        second_closed_edge = cv2.morphologyEx(combined_result, cv2.MORPH_CLOSE, close_kernel)

        return {
            'gray_image': gray_image,
            'white_mask': white_mask,
            'canny_edges': canny_edges,
            'opened_edges': opened_edges,
            'closed_edges': closed_edges,
            'vertical_edges': vertical_edges,
            'enhanced_vertical': enhanced_vertical,
            'combined_result': combined_result,
            'contours': first_contour,
            'license_plates': license_plates,
            'second_closed_edge': second_closed_edge
        }
    
    def draw_license_plates(self, image, results):
        """在图像上绘制检测到的车牌区域"""
        result_image = image.copy()
        
        # 绘制检测到的车牌（绿色矩形框）
        for i, plate in enumerate(results['license_plates']):
            x, y, w, h = plate['x'], plate['y'], plate['width'], plate['height']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # 添加标签
            label = f"Plate {i+1} (AR:{plate['aspect_ratio']:.2f}, Conf:{plate['confidence']:.2f})"
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result_image
    
    def save_detection_results(self, image, results, image_path, output_dir="result"):
        """
        保存检测结果：
        1. 将标记车牌的图片保存到result文件夹
        2. 将车牌坐标信息保存为json文件
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成时间戳作为文件名的一部分
        timestamp = datetime.now().strftime("%m%d_%M%S")
        image_name = os.path.basename(image_path).split('.')[0]
        
        # 绘制结果并保存图片
        result_image = self.draw_license_plates(image, results)
        output_image_path = os.path.join(output_dir, f"{image_name}_{timestamp}.jpg")
        cv2.imwrite(output_image_path, result_image)
        
        # 准备JSON数据
        json_data = {
            "image_path": image_path,
            "timestamp": timestamp,
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "license_plates": []
        }
        
        # 添加每个车牌的信息
        for i, plate in enumerate(results['license_plates']):
            json_data["license_plates"].append({
                "id": i + 1,
                "position": {
                    "x": plate['x'],
                    "y": plate['y'],
                    "width": plate['width'],
                    "height": plate['height']
                },
                "coordinates": [
                    [plate['x'], plate['y']],                   # 左上
                    [plate['x'] + plate['width'], plate['y']],  # 右上
                    [plate['x'] + plate['width'], plate['y'] + plate['height']],  # 右下
                    [plate['x'], plate['y'] + plate['height']]                    # 左下
                ],
                "aspect_ratio": plate['aspect_ratio'],
                "confidence": plate['confidence']
            })
        
        # 保存JSON文件
        output_json_path = os.path.join(output_dir, f"{image_name}_{timestamp}.json")
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=2)
        
        return {
            "output_image": output_image_path,
            "output_json": output_json_path,
            "license_plates_count": len(results['license_plates'])
        }

def main():
    parser = argparse.ArgumentParser(description='车牌定位检测系统')
    parser.add_argument('--image_path', default='examples/car3.jpg', help='输入图像路径')
    parser.add_argument('--low-threshold', type=int, default=50, help='Canny低阈值')
    parser.add_argument('--high-threshold', type=int, default=150, help='Canny高阈值')
    parser.add_argument('--blur-size', type=int, default=5, help='高斯模糊内核大小')
    parser.add_argument('--simple', action='store_true', help='简化模式：只显示最终结果，不使用网格显示')
    parser.add_argument('--no-display', action='store_true', help='不显示任何窗口')
    parser.add_argument('--no-save', action='store_true', help='不保存结果文件')
    
    args = parser.parse_args()
    
    try:
        # 加载图像
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
        print("=== 标记点3: 图像显示===")
        # 执行检测
        print("🔍 正在检测车牌...")
        results = detector.detect_license_plates(image)
        # 显示所有处理步骤
        image_utils.show_images(results, "车牌检测全过程")
        image_utils.show_images(results['white_mask'], "车牌检测结果")

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