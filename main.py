import cv2
import argparse
from plate_detector import LicensePlateDetector
from image_utils import load_image
from color_config import PlateColorConfig

def main():
    parser = argparse.ArgumentParser(description='车牌检测系统')
    parser.add_argument('image_path', help='输入图像路径')
    parser.add_argument('--no-color', action='store_true', help='不使用颜色检测')
    parser.add_argument('--color', choices=['blue', 'yellow', 'green', 'white', 'black'], 
                       help='指定车牌颜色')
    parser.add_argument('--display-masks', action='store_true', help='显示颜色掩码')
    
    args = parser.parse_args()
    
    try:
        # 加载图像
        image = load_image(args.image_path)
        print(f"图像加载成功: {args.image_path}")
        print(f"图像尺寸: {image.shape}")
        
        # 创建plate检测器
        use_color = not args.no_color
        detector = LicensePlateDetector(use_color_detection=use_color, 
                                     specific_color=args.color)
        
        # 检测车牌
        plates = detector.detect(image)
        print(f"检测到 {len(plates)} 个车牌")


        
        # 绘制边界框
        result_image = detector.draw_boxes(image, plates)
        
        # 显示结果
        cv2.imshow('License Plate Detection', result_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if args.display_masks and use_color:
            # 显示颜色掩码
            color_masks = detector.detect_plates_by_multiple_colors(image)
            for color_name, mask in color_masks.items():
                cv2.imshow(f'{color_name} Mask', mask)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()