import cv2 as cv
import numpy as np

if __name__ == '__main__':
    src = cv.imread("coin.jpg")
    cv.imshow("source", src)
    width = src.shape[0]
    height = src.shape[1]
    print(width, height)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # 阈值处理，将图像二值化突出主体
    ret, binary = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
    cv.imshow("bianry", binary)
    kernel = np.ones((3,3), np.uint8)
    #腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    erode = cv.erode(binary, kernel, iterations=3)
    cv.imshow('erode', erode)
    #膨胀操作将使剩余的白色像素扩张并重新增长回去。
    dilate = cv.dilate(erode, kernel, iterations=3)
    cv.imshow('dilate', dilate)

    #在膨胀之后的图中寻找圆的轮廓
    contours, hierarchy =cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    number = len(contours)
    #将图中圆的轮廓标记出来，并将识别的数量打印在图像上
    cv.drawContours(src, contours, -1, (0, 0, 255), 2)
    cv.putText(src, str(len(contours)), (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (100, 200, 255), 1, 3)

    # 显示标记图像
    cv.imshow('result', src)
    cv.waitKey(0)


