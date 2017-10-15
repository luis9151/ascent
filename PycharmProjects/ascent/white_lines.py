import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

dir = 'D:/Luis/Downloads/data_road/training/image_2'


def load_im(num):
    file = 'um_000'
    num = str(num).zfill(3)
    im_rgb = cv2.imread(dir + '/' + file + num + '.png')
    return im_rgb

def rgb_2_bw(im_rgb):
    r = .2989
    g = .5870
    b = .1140
    img = im_rgb[:,:,0]*r + im_rgb[:,:,1]*g + im_rgb[:,:,2]*b
    return np.uint8(img)


def img_pre_process(img):
    edges = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(edges, 150, 200)
    edges = cv2.Sobel(edges, -1, 1, 0)
    cv2.imwrite(dir + '/' + 'canny.png', edges)
    return edges


def Hough_transform(edges, img, thresh, min_ll, max_lg,f):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=thresh, minLineLength=min_ll, maxLineGap=max_lg)
    lanes = np.zeros(img.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lanes, (x1, y1), (x2, y2), (255), 2)
    cv2.imwrite(dir + '/' + f +'.png', lanes)
    return np.uint8(lanes)


def draw_template(img):
    center = [int(x/2) for x in img.shape]
    corner_left = (img.shape[0],0)
    corner_right = img.shape
    corners = (corner_left,corner_right)
    template = np.zeros(img.shape)
    for corner in corners:
        o1 = 25
        o2 = 400
        if corner[1]<center[1]:
            o1 = o1*-1
            o2 = o2 * -1
        x1 = center[1] + o1
        x2 = corner[1] - o2
        y1 = center[0]
        y2 = corner[0]
        cv2.line(template, (x1, y1), (x2, y2), (255), 2)
    template = np.uint8(template[center[0]:img.shape[0],corner_left[1]+abs(o2):corner_right[1]-abs(o2)+1])
    cv2.imwrite(dir + '/' + 'template' + '.png', template)
    return template


def templ_matching(lanes,template,img):
    h,w = template.shape
    meth = 'cv2.TM_CCOEFF'
    method = eval(meth)
    # Apply template Matching
    res = cv2.matchTemplate(lanes, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (255), 2)
    cv2.imwrite(dir + '/' + 'find_lane' + '.png', img)


def main():
    im_rgb = load_im(10)
    img = rgb_2_bw(im_rgb)
    edges = img_pre_process(img)
    lanes = Hough_transform(edges, img,80,20,15,'lanes')
    #bounds = Hough_transform(edges, img,200,1,50,'bounds')
    template = draw_template(img)
    templ_matching(lanes,template,img)






main()
plt.show()