# coding=utf-8
import cv2
import os
import numpy as np

class ConnectedComponentsApp:
    def save_component_info(self, image_path, num_labels, labels, stats, centroids, area_threshold, total_area):
        """
        保存筛选后的连通组件信息到txt文件
        """
        # 创建test-txt文件夹（如果不存在）
        os.makedirs("test-txt", exist_ok=True)
        
        # 提取输入图像的文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join("test-txt", f"{filename}.txt")
        
        # 收集符合条件的连通组件信息
        component_info = []
        for i in range(num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 跳过背景和面积小于阈值的组件
            if i == 0 or area < area_threshold:
                continue  
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx = centroids[i, 0]
            cy = centroids[i, 1]
            area_ratio = (area / total_area) * 100
            
            # 计算填充率并进行第二次判断
            bounding_area = w * h
            fill_ratio = (area / bounding_area) * 100
            if fill_ratio >= 15:
                component_info.append({
                    'id': i,
                    'area': area,
                    'area_ratio': area_ratio,
                    'fill_ratio': fill_ratio,
                    'bounding_box': (x, y, w, h),
                    'centroid': (cx, cy)
                })
        
        # 写入txt文件
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("筛选后的连通组件信息（面积≥0.2%且填充率≥15%）\n")
            f.write("=" * 80 + "\n")
            f.write(f"图像文件名：{os.path.basename(image_path)}\n")
            f.write(f"总筛选后的组件数量：{len(component_info)}\n")
            f.write(f"图像总面积：{total_area} 像素\n")
            f.write(f"面积阈值（0.2%）：{area_threshold:.0f} 像素\n")
            f.write("=" * 80 + "\n")
            f.write("组件ID | 面积(像素) | 占比(%) | 填充率(%) | 边界框(x, y, w, h) | 质心坐标(cx, cy)\n")
            f.write("-" * 105 + "\n")
            for info in component_info:
                f.write(f"{info['id']:4d} | {info['area']:10d} | {info['area_ratio']:5.1f}% | {info['fill_ratio']:5.1f}% | ({info['bounding_box'][0]:4d}, {info['bounding_box'][1]:4d}, {info['bounding_box'][2]:4d}, {info['bounding_box'][3]:4d}) | ({info['centroid'][0]:6.1f}, {info['centroid'][1]:6.1f})\n")
            f.write("-" * 105 + "\n")
        
        print(f"\n连通组件信息已保存到：{txt_path}")
    
def process_image(image_path):
    """
    检测连通组件并进行标记
    """
    # 加载二值化图片
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"无法加载图片: {image_path}")
        return
    
    # 确保图像是二值化的
    _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
    
    # 检测连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)
    
    # 计算图像总面积和0.2%阈值
    image_height, image_width = binary_image.shape
    total_area = image_height * image_width
    area_threshold = total_area * 0.001 # 0.2%的面积阈值
    # 统计符合条件的物体数量
    valid_objects = 0
    for i in range(1, num_labels):  # 从1开始，不包括背景
        # 第一次判断：面积大于0.2%
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            # 第二次判断：标记的图像面积占连通矩阵面积的15%以上
            component_area = stats[i, cv2.CC_STAT_AREA]
            bounding_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
            if component_area / bounding_area >= 0.15:
                valid_objects += 1
    
    
    # 打印连通组件检测结果
    print(f"\n处理图片: {os.path.basename(image_path)}")
    print("连通组件检测结果：")
    print(f"总连通组件数量（包括背景）：{num_labels}")
    print(f"实际物体数量（不包括背景）：{num_labels - 1}")
    print(f"面积大于0.2%的物体数量：{valid_objects}")
    print(f"图像总面积：{total_area} 像素")
    print(f"面积阈值（0.2%）：{area_threshold:.0f} 像素")
    
    # 打印符合条件的连通组件信息
    print("\n面积大于0.2%且填充率大于15%的连通组件信息：")
    print("组件ID | 面积(像素) | 占比(%) | 填充率(%) | 边界框(x, y, w, h) | 质心坐标(cx, cy)")
    print("-" * 105)
    for i in range(num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # 跳过背景和面积小于阈值的组件
        if i == 0 or area < area_threshold:
            continue  
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx = centroids[i, 0]
        cy = centroids[i, 1]
        area_ratio = (area / total_area) * 100
        
        # 计算填充率并进行第二次判断
        bounding_area = w * h
        fill_ratio = (area / bounding_area) * 100
        if fill_ratio >= 15:
            print(f"{i:4d} | {area:10d} | {area_ratio:5.1f}% | {fill_ratio:5.1f}% | ({x:4d}, {y:4d}, {w:4d}, {h:4d}) | ({cx:6.1f}, {cy:6.1f})")
    print("-" * 105)
    
    # 保存连通组件信息到txt文件
    app = ConnectedComponentsApp()
    app.save_component_info(image_path, num_labels, labels, stats, centroids, area_threshold, total_area)
    
    # 生成随机颜色用于标记不同的连通组  要改
    colors = []
    for i in range(num_labels):
        # 背景用黑色
        if i == 0:
            colors.append([0, 0, 0])
        else:
            # 第一次判断：面积大于0.2%
            if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                # 第二次判断：填充率大于15%
                component_area = stats[i, cv2.CC_STAT_AREA]
                bounding_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
                if component_area / bounding_area >= 0.15:
                    colors.append([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
                else:
                    colors.append([0, 0, 0])  # 使用背景色（不标记）
            else:
                colors.append([0, 0, 0])  # 使用背景色（不标记）
    colors = np.array(colors, dtype=np.uint8)
    
    # 创建标记后的彩色图像
    labeled_colors = colors[labels]
    
    # 在标记后的图像上绘制组件边界（仅对面积大于0.2%且填充率大于15%的组件）
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            # 第二次判断：填充率大于15%
            component_area = stats[i, cv2.CC_STAT_AREA]
            bounding_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
            if component_area / bounding_area >= 0.15:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(labeled_colors, (x, y), (x+w, y+h), (255, 255, 255), 2)
    
    # 保存标记后的图像到examples-connect文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_connect_dir = os.path.join(current_dir, "examples-connect")
    os.makedirs(examples_connect_dir, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_labeled.jpg"
    output_path = os.path.join(examples_connect_dir, output_filename)
    cv2.imwrite(output_path, labeled_colors)
    

def process_batch():
    """
    批量处理examples-binaty文件夹中的所有图片
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置examples-binary文件夹路径
    input_dir = os.path.join(current_dir, "examples-binary")
    
    # 检查文件夹是否存在
    if not os.path.exists(input_dir):
        print(f"文件夹不存在: {input_dir}")
        print("请确保examples-binary文件夹存在且包含二值化图片")
        return
    
    # 创建test-txt文件夹（如果不存在）
    output_dir = os.path.join(current_dir, "test-txt")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件夹中的所有图片文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        print(f"在{input_dir}中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始批量处理...")
    
    # 对每张图片进行处理
    for image_path in image_files:
        try:
            process_image(image_path)
        except Exception as e:
            print(f"处理图片 {os.path.basename(image_path)} 时出错: {str(e)}")
    
    print("\n批量处理完成！")

def main():
    process_batch()

if __name__ == "__main__":
    main()