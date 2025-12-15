# coding=utf-8
import cv2
import numpy as np
def merge_lines(lines, angle_thresh=8, dist_thresh=15):
    """
    合并共线且相邻的线段
    
    lines: 线段列表，每条线段表示为 [x1, y1, x2, y2]
    angle_thresh: 角度阈值（度）
    dist_thresh: 距离阈值（像素）
    """

    merged_lines = []
    used = [False] * len(lines)
    
    for dline in lines:
        x1 = int(round(dline[0][0]))
        y1 = int(round(dline[0][1]))
        x2 = int(round(dline[0][2]))
        y2 = int(round(dline[0][3]))
        current_group = [i]
        used[i] = True
        
        # 计算当前线段的角度和长度
        angle_i = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        length_i = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 查找可合并的线段
        for j in range(i+1, len(lines)):
            if used[j]:
                continue
                
            x3, y3, x4, y4 = lines[j]
            angle_j = np.degrees(np.arctan2(y4 - y3, x4 - x3))
            
            # 角度差（考虑角度周期性）
            angle_diff = min(abs(angle_i - angle_j), 
                           abs(angle_i - angle_j - 180),
                           abs(angle_i - angle_j + 180))
            
            if angle_diff < angle_thresh:
                # 计算线段之间的距离
                dist = min_distance_between_lines(lines[i], lines[j])
                if dist < dist_thresh:
                    current_group.append(j)
                    used[j] = True
        
        # 合并组内的线段
        if len(current_group) > 1:
            merged_line = merge_line_group(lines, current_group)
            merged_lines.append(merged_line)
        else:
            merged_lines.append(lines[i])
    
    return merged_lines

def process_license_plate_detection(img, fld):
    """
    完整的车牌区域检测流程（修正版）
    """
    # 1. 检测线段
    dlines = fld.detect(img)
    if dlines is None or len(dlines) == 0:
        print("未检测到线段")
        return []
    
    # 2. 正确提取线段坐标
    lines = []
    for dline in dlines:
        # 这里是你提到的方式
        x1 = int(round(dline[0][0]))
        y1 = int(round(dline[0][1]))
        x2 = int(round(dline[0][2]))
        y2 = int(round(dline[0][3]))
        lines.append([x1, y1, x2, y2])
    
    print(f"原始检测到 {len(lines)} 条线段")
    
    # 3. 可视化原始线段
    img_with_lines = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 4. 线段过滤和合并
    filtered_lines = filter_lines(lines)
    merged_lines = merge_similar_lines(filtered_lines)
    
    print(f"过滤合并后剩下 {len(merged_lines)} 条线段")
    
    return merged_lines, img_with_lines

def filter_lines(lines, min_length=20, angle_tolerance=20):
    """
    过滤线段：长度和角度过滤
    """
    filtered = []
    
    for x1, y1, x2, y2 in lines:
        # 计算长度
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if length < min_length:
            continue
        
        # 计算角度（0-180度）
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        # 只保留接近水平或垂直的线段
        # 水平：0° 或 180°，垂直：90°
        is_horizontal = (angle < angle_tolerance or 
                        angle > 180 - angle_tolerance)
        is_vertical = (abs(angle - 90) < angle_tolerance)
        
        if is_horizontal or is_vertical:
            filtered.append([x1, y1, x2, y2])
    
    return filtered

def merge_similar_lines(lines, angle_thresh=10, dist_thresh=20):
    """
    合并相似线段
    """
    if not lines:
        return []
    
    # 计算每条线段的角度和长度
    line_info = []
    for x1, y1, x2, y2 in lines:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        line_info.append({
            'coords': (x1, y1, x2, y2),
            'angle': angle,
            'length': length,
            'center': (center_x, center_y)
        })
    
    # 按角度分组
    groups = []
    used = [False] * len(line_info)
    
    for i in range(len(line_info)):
        if used[i]:
            continue
        
        # 创建新组
        current_group = [i]
        used[i] = True
        angle_i = line_info[i]['angle']
        
        # 寻找相似线段
        for j in range(i + 1, len(line_info)):
            if used[j]:
                continue
            
            angle_j = line_info[j]['angle']
            angle_diff = min(abs(angle_i - angle_j),
                           abs(angle_i - angle_j - 180),
                           abs(angle_i - angle_j + 180))
            
            if angle_diff < angle_thresh:
                # 检查距离
                dist = distance_between_lines(line_info[i], line_info[j])
                if dist < dist_thresh:
                    current_group.append(j)
                    used[j] = True
        
        groups.append(current_group)
    
    # 合并每组内的线段
    merged_lines = []
    for group in groups:
        if len(group) == 1:
            # 单个线段，直接保留
            x1, y1, x2, y2 = line_info[group[0]]['coords']
            merged_lines.append([x1, y1, x2, y2])
        else:
            # 合并多个线段
            merged_line = merge_line_group([line_info[i] for i in group])
            merged_lines.append(merged_line)
    
    return merged_lines

def distance_between_lines(line1, line2):
    """
    计算两条线段之间的距离
    """
    # 计算两条线段中点的距离
    cx1, cy1 = line1['center']
    cx2, cy2 = line2['center']
    return np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

def merge_line_group(line_group):
    """
    合并一组线段
    """
    if not line_group:
        return None
    
    # 计算平均角度
    avg_angle = np.mean([line['angle'] for line in line_group])
    
    # 收集所有端点
    all_points = []
    for line in line_group:
        x1, y1, x2, y2 = line['coords']
        all_points.append((x1, y1))
        all_points.append((x2, y2))
    
    # 在平均角度方向上投影所有点
    angle_rad = np.radians(avg_angle)
    
    # 沿角度方向的单位向量
    ux = np.cos(angle_rad)
    uy = np.sin(angle_rad)
    
    # 计算所有点在角度方向上的投影值
    projections = []
    for x, y in all_points:
        projection = x * ux + y * uy
        projections.append(projection)
    
    # 找到最小和最大投影值对应的点
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    
    x1, y1 = all_points[min_idx]
    x2, y2 = all_points[max_idx]
    
    return [int(x1), int(y1), int(x2), int(y2)]
# 主程序流程示例
def main():
    # 读取图像
    img = cv2.imread('./examples/car3.jpg')
    if img is None:
        print("无法读取图像")
        return
    
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建LSD检测器
    fld = cv2.ximgproc.createFastLineDetector()
    
    # 检测车牌区域
    merged_lines, img_with_lines = process_license_plate_detection(gray, fld)
    
    # 可视化结果
    result_img = img.copy()
    
    # 绘制合并后的线段
    for x1, y1, x2, y2 in merged_lines:
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制检测到的四边形
    # for i, quad in enumerate(quads):
    #     # quad是四个点的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    #     pts = np.array(quad, dtype=np.int32)
    #     cv2.polylines(result_img, [pts], True, (0, 0, 255), 3)
    
    # 显示结果
    cv2.imshow("Original Lines", img_with_lines)
    cv2.imshow("Processed Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return merged_lines
if __name__ == "__main__":
    main()
