import cv2
import numpy as np
from pathlib import Path

class PlateLocator:
    def __init__(self):
        pass

    # 预处理 + 白底黑字增强（保留你原有逻辑）
    def preprocess_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        mask_bgr = cv2.inRange(image, np.array([200, 200, 200]), np.array([255, 255, 255]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_thresh = max(180, int(np.mean(gray)+20))
        _, mask_gray = cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(mask_hsv, mask_bgr)
        mask = cv2.bitwise_or(mask, mask_gray)

        kernel = np.ones((3,3), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (19,7))
        mask_final = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_large)
        return mask_final

    # 计算特征（原始函数保留）
    def calculate_plate_features(self, contour, image_area):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w/h if h>0 else 0
        area_ratio = area / image_area
        perimeter = cv2.arcLength(contour, True)
        compactness = 4*np.pi*area/(perimeter*perimeter) if perimeter>0 else 0
        rect_ratio = area/(w*h) if (w*h)>0 else 0
        return {'bbox':(x,y,w,h),'area':area,'aspect_ratio':aspect_ratio,
                'area_ratio':area_ratio,'compactness':compactness,'rect_ratio':rect_ratio,
                'width':w,'height':h}

    # 评分函数
    def score_plate_candidate(self, features):
        score = 0
        ar = features['aspect_ratio']
        if 1.8 <= ar <= 2.5: score +=3
        elif 1.5 <= ar <=3.0: score +=2
        elif 1.3 <= ar <=4.0: score +=1
        if 0.005 <= features['area_ratio'] <=0.15: score +=2
        elif 0.001 <= features['area_ratio'] <=0.20: score+=1
        if features['rect_ratio']>0.7: score +=2
        elif features['rect_ratio']>0.5: score +=1
        if 0.65 <= features['compactness']<=0.9: score +=1.5
        w,h = features['width'],features['height']
        if 50<=w<=400 and 20<=h<=200: score+=1.5
        elif 30<=w<=500 and 15<=h<=250: score+=1
        return score

    # 倾斜校正
    def correct_skew(self, image):
        if image is None: return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image
        rect = cv2.minAreaRect(max(contours,key=cv2.contourArea))
        angle = rect[2]
        if angle < -45: angle = 90 + angle
        elif angle >45: angle = angle - 90
        if abs(angle)>1.5:
            h,w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
            return cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image

    # 主定位接口
    def locate_plate(self, img):
        mask = self.preprocess_image(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img.shape[0]*img.shape[1]
        candidates = []
        for cnt in contours:
            f = self.calculate_plate_features(cnt, img_area)
            if f['area']>800 and 1.2<f['aspect_ratio']<6.0:
                score = self.score_plate_candidate(f)
                candidates.append({'score':score,'bbox':f['bbox']})
        if not candidates: return None
        best = max(candidates,key=lambda x:x['score'])
        x,y,w,h = best['bbox']
        px, py = int(w*0.1), int(h*0.1)
        roi = img[max(0,y-py):y+h+py, max(0,x-px):x+w+px]
        corrected = self.correct_skew(roi)
        return corrected

if __name__ == "__main__":
    import sys
    import os
    import tkinter as tk
    from tkinter import filedialog
    
    # 创建Tkinter根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 设置默认打开路径为examples文件夹
    default_dir = os.path.join(os.path.dirname(__file__), 'examples')
    if not os.path.exists(default_dir):
        default_dir = os.path.dirname(__file__)
    
    # 打开文件选择对话框
    image_path = filedialog.askopenfilename(
        title="选择图片文件",
        initialdir=default_dir,
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not image_path:
        print("未选择图片")
        sys.exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)
    locator = PlateLocator()
    roi = locator.locate_plate(img)
    if roi is not None:
        cv2.imwrite("cropped_plate.png", roi)
        print("定位完成，保存到 cropped_plate.png")
