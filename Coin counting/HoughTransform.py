import cv2 as cv
import numpy as np

if __name__ == '__main__':
    src = cv.imread("coin.jpg")
    cv.imshow("source", src)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #阈值处理，将图像二值化突出主体
    ret,img = cv.threshold(src, 100, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # 霍夫变换圆检测
    #第四个参数为圆心之间的最小距离
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=120, param2=40, minRadius=5, maxRadius=80)
    # 输出检测到圆的个数
    print(len(circles[0]))

    i = 1
    # 根据检测到圆的信息，画出每一个圆
    for circle in circles[0]:
        # 圆的基本信息
        #print(circle[2])
        # 坐标行列
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 在原图用指定颜色标记出圆的位置
        img = cv.circle(src, (x, y), r, (0, 0, 255), -1)
        cv.putText(src, str(i), (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (100, 200, 255), 2, 8)
        i = i + 1
    # 显示标记图像
    cv.imshow('result', img)
    cv.waitKey(0)


