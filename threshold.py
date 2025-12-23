import cv2
phone = cv2.imread('car1.jpg')#读取原图
phone_gray = cv2.cvtColor(phone,cv2.COLOR_BGR2GRAY)#灰度图的处理
cv2.imshow('phone_b',phone_gray)
cv2.waitKey(0)
# phone_gray=cv2.imread('phone.png',0)  #读取灰度图
ret, phone_binary = cv2.threshold(phone_gray, 120, 255, cv2.THRESH_BINARY)#阈值处理为二值
cv2.imshow('phone_binary',phone_binary)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(phone_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)