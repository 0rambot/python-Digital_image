import cv2 as cv
import numpy as np
if __name__ == '__main__':
    src = cv.imread("circle.jpeg")
    #cv.imshow("source",src)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)
    #cv.imshow("binary", binary)

    kernel = np.ones((12, 12), np.uint8)
    closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("closing", closing)

    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("opening", opening)

    # 腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    kernel1 = np.ones((7,7), np.uint8)
    erode = cv.erode(opening, kernel1, iterations=6)
    cv.imshow('erode', erode)
    # 膨胀操作将使剩余的白色像素扩张并重新增长回去。
    dilate = cv.dilate(erode, kernel1, iterations=6)
    cv.imshow('dilate', dilate)
    kernel2 = np.ones((12,12), np.uint8)
    median = cv.medianBlur(dilate, 91)
    cv.imshow("result3",median)

    # 在膨胀之后的图中寻找圆的轮廓
    contours, hierarchy = cv.findContours(median, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 将轮廓标记出来
    cv.drawContours(src, contours, -1, (0, 0, 255), 2)

    cv.imshow("result", src)
    cv.waitKey(0)
