import numpy as np
import cv2 as cv
roi = cv.imread('sample1.png')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target = cv.imread('bluecar.jpg')
cv.imshow("source",target)
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

# 计算模型样例的直方图
roihist = cv.calcHist([hsv],[0,1], None, [160,250], [0,160,0,250] )
# 直方图归一化并利用反传算法
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,160,0,250],1)

# 进行卷积
disc = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
cv.filter2D(dst,-1,disc,dst)


# 应用阈值作与操作
ret,thresh = cv.threshold(dst,50,240,0)
thresh = cv.merge((thresh,thresh,thresh))

# 腐蚀
kernel1 = np.ones((11,11),np.uint8)
erosion = cv.erode(thresh,kernel1,iterations = 1)

# 膨胀
kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(12,12))
dilate = cv.dilate(erosion,kernel2)

result = cv.bitwise_and(target,dilate)
cv.imshow('result',result)

cv.waitKey(0)
