# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

def recognize_chars(char_imgs):
    """
    字符识别函数，对应MATLAB的recognize_chars.m
    功能：将分割后的字符与模板库进行匹配，识别出对应的字符
    参数：char_imgs - 分割后的字符图像列表
    返回：char_result - 识别结果字符串
    """
    char_result = ""
    
    # 模板文件夹路径
    template_folder = os.path.join(os.getcwd(), '字符模板')
    
    # ================== 汉字识别（第一个字符） ==================
    chinese_chars = ['藏','川','鄂','甘','赣','港','桂','贵','黑','沪','吉','京','津','晋','辽','鲁','蒙','闽','青','琼','陕','苏','台','皖','湘','新','渝','豫','粤','云','浙']
    
    if len(char_imgs) > 0:
        Char_1 = char_imgs[0]
        
        if Char_1 is not None and np.any(Char_1 > 0):  # 确保不是全空
            best_score = float('inf')
            best_char = '?'
            
            for j in range(len(chinese_chars)):
                char_name = chinese_chars[j]
                template_path = os.path.join(template_folder, f'{char_name}.bmp')
                
                if os.path.exists(template_path):
                    try:
                        # 使用绝对路径并确保正确的编码
                        template_path_abs = os.path.abspath(template_path)
                        Template = cv2.imdecode(np.fromfile(template_path_abs, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        if Template is None:
                            continue
                        
                        if len(Template.shape) == 3:
                            Template = cv2.cvtColor(Template, cv2.COLOR_BGR2GRAY)
                        
                        # 二值化模板
                        _, Template_bw = cv2.threshold(Template, 127, 255, cv2.THRESH_BINARY)
                        
                        # 调整模板尺寸
                        Template_resized = cv2.resize(Template_bw, (16, 32), interpolation=cv2.INTER_NEAREST)
                        
                        if Char_1.shape == Template_resized.shape:
                            # 确保字符图像是二值化的
                            if len(Char_1.shape) == 3:
                                Char_1_gray = cv2.cvtColor(Char_1, cv2.COLOR_BGR2GRAY)
                                _, Char_1_bw = cv2.threshold(Char_1_gray, 127, 255, cv2.THRESH_BINARY)
                            elif len(Char_1.shape) == 2 and Char_1.dtype != np.uint8:
                                Char_1_gray = cv2.normalize(Char_1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                _, Char_1_bw = cv2.threshold(Char_1_gray, 127, 255, cv2.THRESH_BINARY)
                            else:
                                _, Char_1_bw = cv2.threshold(Char_1, 127, 255, cv2.THRESH_BINARY)
                            
                            # 计算差异
                            Differ = cv2.absdiff(Char_1_bw, Template_resized)
                            score = np.sum(Differ)
                            
                            if score < best_score:
                                best_score = score
                                best_char = char_name
                    except Exception as e:
                        continue
            char_result += best_char
        else:
            char_result += '?'
    
    # ================== 字母数字识别（第2-7个字符） ==================
    alphanum_chars = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    for i in range(1, 7):  # 第2到第7个字符
        if i < len(char_imgs):
            char_img = char_imgs[i]
        else:
            char_img = None
        
        if char_img is not None and np.any(char_img > 0):
            best_score = float('inf')
            best_char = '?'
            
            for j in range(len(alphanum_chars)):
                char_name = alphanum_chars[j]
                template_path = os.path.join(template_folder, f'{char_name}.bmp')
                
                if os.path.exists(template_path):
                    try:
                        # 使用绝对路径并确保正确的编码
                        template_path_abs = os.path.abspath(template_path)
                        Template = cv2.imdecode(np.fromfile(template_path_abs, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        if Template is None:
                            continue
                        
                        if len(Template.shape) == 3:
                            Template = cv2.cvtColor(Template, cv2.COLOR_BGR2GRAY)
                        
                        # 二值化模板
                        _, Template_bw = cv2.threshold(Template, 127, 255, cv2.THRESH_BINARY)
                        
                        # 调整模板尺寸
                        Template_resized = cv2.resize(Template_bw, (16, 32), interpolation=cv2.INTER_NEAREST)
                        
                        if char_img.shape == Template_resized.shape:
                            # 确保字符图像是二值化的
                            if len(char_img.shape) == 3:
                                char_img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
                                _, char_img_bw = cv2.threshold(char_img_gray, 127, 255, cv2.THRESH_BINARY)
                            elif len(char_img.shape) == 2 and char_img.dtype != np.uint8:
                                char_img_gray = cv2.normalize(char_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                _, char_img_bw = cv2.threshold(char_img_gray, 127, 255, cv2.THRESH_BINARY)
                            else:
                                _, char_img_bw = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
                            
                            # 计算差异
                            Differ = cv2.absdiff(char_img_bw, Template_resized)
                            score = np.sum(Differ)
                            
                            if score < best_score:
                                best_score = score
                                best_char = char_name
                    except Exception as e:
                        continue
            char_result += best_char
        else:
            char_result += '?'
    
    return char_result