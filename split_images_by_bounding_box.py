# coding=utf-8
import os
import cv2
import glob

class ImageSplitter:
    def __init__(self):
        self.txt_folder = "test-txt"
        self.image_folder = "examples"
        self.output_folder = "test"
        
        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)
    
    def parse_txt_file(self, txt_path):
        """
        解析txt文件，提取边界框信息
        """
        components = []
        
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 找到表头行
                header_found = False
                for i, line in enumerate(lines):
                    line = line.strip()
                    if "组件ID | 面积(像素) | 占比(%) | 填充率(%) | 边界框(x, y, w, h) | 质心坐标(cx, cy)" in line:
                        header_found = True
                        # 从下一行开始读取数据
                        for data_line in lines[i+2:]:  # 跳过表头和分隔线
                            data_line = data_line.strip()
                            if data_line and "-" not in data_line:
                                # 解析数据行
                                parts = data_line.split("|")
                                if len(parts) >= 6:
                                    try:
                                        component_id = int(parts[0].strip())
                                        # 解析边界框信息
                                        bbox_str = parts[4].strip()
                                        # 去除括号
                                        bbox_str = bbox_str[1:-1].strip()
                                        # 分割坐标和尺寸
                                        bbox_parts = bbox_str.split(",")
                                        if len(bbox_parts) == 4:
                                            x = int(bbox_parts[0].strip())
                                            y = int(bbox_parts[1].strip())
                                            w = int(bbox_parts[2].strip())
                                            h = int(bbox_parts[3].strip())
                                            components.append({
                                                'id': component_id,
                                                'x': x,
                                                'y': y,
                                                'w': w,
                                                'h': h
                                            })
                                    except (ValueError, IndexError):
                                        continue
                return components
        except Exception as e:
            print(f"解析txt文件 {txt_path} 时出错: {e}")
            return []
    
    def split_images(self):
        """
        批量处理所有图像文件
        """
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(self.txt_folder, "*.txt"))
        
        for txt_file in txt_files:
            # 提取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(txt_file))[0]
            
            # 对应的图像文件路径
            image_path = os.path.join(self.image_folder, f"{filename}.jpg")
            
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"图像文件 {image_path} 不存在，跳过处理")
                continue
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"读取图像 {image_path} 失败，跳过处理")
                continue
            
            # 解析txt文件获取边界框信息
            components = self.parse_txt_file(txt_file)
            
            if not components:
                print(f"在 {txt_file} 中未找到有效的边界框信息，跳过处理")
                continue
            
            print(f"处理图像 {image_path}，找到 {len(components)} 个组件")
            
            # 根据边界框分割图像
            for component in components:
                component_id = component['id']
                x = component['x']
                y = component['y']
                w = component['w']
                h = component['h']
                
                # 确保边界框在图像范围内
                image_height, image_width = image.shape[:2]
                x = max(0, x)
                y = max(0, y)
                x2 = min(image_width, x + w)
                y2 = min(image_height, y + h)
                
                # 分割图像
                cropped_image = image[y:y2, x:x2]
                
                # 保存分割后的图像
                output_filename = f"{filename}_{component_id}.png"
                output_path = os.path.join(self.output_folder, output_filename)
                
                if cv2.imwrite(output_path, cropped_image):
                    print(f"保存分割后的图像: {output_path}")
                else:
                    print(f"保存分割后的图像 {output_path} 失败")

if __name__ == "__main__":
    splitter = ImageSplitter()
    splitter.split_images()
