import numpy as np
import os
import cv2
from matplotlib import pyplot as plt


def load_im():
    dir = 'D:/Luis/Downloads/data_road/training/image_2'
    file = 'um_0000'
    num = str(1).zfill(2)
    im_rgb = cv2.imread(dir + '/' + file + num + '.png')
    return im_rgb

def rgb_2_bw(im_rgb):
    r = .2989
    g = .5870
    b = .1140
    im = im_rgb[:,:,0]*r + im_rgb[:,:,1]*g + im_rgb[:,:,2]*b
    return np.uint8(im)


def main():
    dir = 'D:/Luis/Downloads/data_road/training/image_2'
    im_rgb = load_im()
    img = rgb_2_bw(im_rgb)

    edges = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(edges,150,200)
    edges = cv2.Sobel(edges,-1,1,0)
    cv2.imwrite(dir + '/' + 'canny.png', edges)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=20, maxLineGap=15)
    lanes = np.zeros(img.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #cv2.line(im_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(lanes, (x1, y1), (x2, y2),(255), 2)
    cv2.imwrite(dir + '/' + 'houghlines5.png', lanes)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=200, minLineLength=1, maxLineGap=50)
    bounds = np.zeros(img.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(bounds, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imwrite(dir + '/' + 'houghlines5.png', bounds)

    #dir = 'D:/Luis/Downloads/data_road/training/image_2'
    #cv2.imwrite(dir + '/' + 'ex' + '.png', im)
    a = 1



main()