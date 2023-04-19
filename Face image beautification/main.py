import cv2 as cv
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

def imgConvolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h+1, j - padding_w:j + padding_w+1]*kernel))

    return image_convolve

# Prewitt Edge
def prewittEdge(image, prewitt_x, prewitt_y):
    '''
    :param image: 图片矩阵
    :param prewitt_x: 竖直方向
    :param prewitt_y:  水平方向
    :return:处理后的矩阵
    '''
    img_X = imgConvolve(image, prewitt_x)
    img_Y = imgConvolve(image, prewitt_y)

    img_prediction = np.zeros(img_X.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_X[i][j], img_Y[i][j])
    return img_prediction

# LoG filter
def LoG_filter(img, K_size=5, sigma=3):
    H, W = img.shape

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    # LoG Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

def main():
    src = cv.imread("picture1.jpg");
    cv.imshow("source",src)
    bilateral = cv.bilateralFilter(src, 43, 120, 15);
    cv.imshow("bilateral", bilateral)

    hsv = cv.cvtColor(bilateral, cv.COLOR_BGR2HSV)
    (H, S, V) = cv.split(hsv)

    kernel = np.array([[0, 1, 0], [-1, 4, -1], [0, 1, 0]])
    kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    laplacian = cv.filter2D(V,-1,kernel2)
    cv.imshow("laplacian", laplacian)

    #out = LoG_filter(V, 5, 3)
    #cv.imshow("log", out)

    prewitt_1 = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_2 = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])
    img_prewitt1 = prewittEdge(V, prewitt_1, prewitt_2)
    print(V.shape, img_prewitt1.shape)
    cv.imshow('prewitt', img_prewitt1)

    sobel_1 = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_2 = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    sobel = imgConvolve(V, sobel_1)

    #V = cv.subtract(V, img_prewitt1, cv.CV_32F)
    cv.imshow("substract", sobel)

    #V = cv.add(V, img_prewitt1, int)

    merge = cv.merge([H, S, V])
    result = cv.cvtColor(merge, cv.COLOR_HSV2BGR)

    cv.imshow("result", result)

    """
    hsv = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    (H,S,V) = cv.split(hsv)
    cv.imshow("H", H)
    cv.imshow("S", S)
    cv.imshow("V", V)
    """

    """
    #伽马变换
    addimg = addimg/255
    gamma = 0.8
    result = np.power(addimg,gamma)
    cv.imshow("result", result)
    """
    cv.waitKey(0);

if __name__  ==  "__main__":
    main()

