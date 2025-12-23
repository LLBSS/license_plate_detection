import cv2
import os
import glob

# 创建保存结果的文件夹
output_folder = 'examples-canny'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义目标最大尺寸
max_width = 800
max_height = 600

# 遍历examples文件夹中的所有图片文件
image_files = cv2.imread('car1.jpg')
# 获取原始图像尺寸
height, width = image_files.shape[:2]

# 计算缩放比例
scale = min(max_width / width, max_height / height)

# 调整图像大小
if scale < 1:
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = cv2.resize(image_files, (new_width, new_height), interpolation=cv2.INTER_AREA)
else:
    img_resized = image_files

# 进行Canny边缘检测
img_canny = cv2.Canny(img_resized, 100, 150)  # 低阈值，高阈值

# 保存Canny边缘检测结果
cv2.imwrite('car1-canny.jpg', img_canny)
print("所有图像处理完成！")