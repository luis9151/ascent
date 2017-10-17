import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import glob


def set_dirs(f,which_dir):
    global dir_read
    dir_read = '{}/data_road/{}/image_2/'.format(f,which_dir)
    global dir_write
    dir_write = dir_read+'results/'
    if not os.path.isdir(dir_write):
        os.mkdir(dir_write)


def rgb_2_bw(im_rgb):
    r = .2989
    g = .5870
    b = .1140
    img = im_rgb[:,:,0]*r + im_rgb[:,:,1]*g + im_rgb[:,:,2]*b
    return np.uint8(img)


def img_pre_process(edges):
    k = 9
    k2 = 3
    edges = cv2.GaussianBlur(edges, (k, k), 0)
    edges = cv2.Sobel(edges, -1, k2-2, 0, ksize=k2)
    edges = cv2.Canny(edges, 150, 200,L2gradient=False)
    return edges


def Hough_transform(edges, img, thresh, min_ll, max_lg,n):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=thresh, minLineLength=min_ll, maxLineGap=max_lg)
    lanes = np.zeros(img.shape)
    if lines is None:
        print('No lanes found for image #{}'.format(n))
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lanes, (x1, y1), (x2, y2), (255), 2)
    return np.uint8(lanes)


def draw_template(img,offset1,offset2):
    center = [int(x/2) for x in img.shape]
    corner_left = (img.shape[0],0)
    corner_right = img.shape
    corners = (corner_left,corner_right)
    template = np.zeros(img.shape)
    for corner in corners:
        o1 = offset1
        o2 = offset2
        if corner[1]<center[1]:
            o1 = o1*-1
            o2 = o2 * -1
        x1 = center[1] + o1
        x2 = corner[1] - o2
        y1 = center[0]
        y2 = corner[0]
        cv2.line(template, (x1, y1), (x2, y2), (255), 2)
    template = np.uint8(template[center[0]:img.shape[0],corner_left[1]+abs(o2):corner_right[1]-abs(o2)+1])
    return template


def templ_matching(lanes,template,img):
    center = [int(x/2) for x in img.shape]
    lanes2 = lanes[center[0]:,:]
    h,w = template.shape
    meth = 'cv2.TM_CCOEFF'
    method = eval(meth)
    res = cv2.matchTemplate(lanes2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    top_left = (top_left[0], top_left[1] + center[0])
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img = img.copy()
    cv2.rectangle(img, top_left, bottom_right, (255), 2)
    return lanes[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]], (top_left,bottom_right)


def align_lanes(img,edges,xy,n):
    highlight = np.zeros(img[:,:,0].shape)
    highlight[xy[0][1]:xy[1][1],xy[0][0]:xy[1][0]] = edges
    img[:, :, 0] = np.clip(img[:,:,0] - highlight*255, 0, None)
    img[:, :, 1] = np.clip(img[:,:,1] - highlight*255, 0, None)
    img[:, :, 2] = np.clip(img[:,:,2] + highlight*255, None, 255)
    cv2.imwrite(dir_write + '/' + 'final_img_' + str(n).zfill(3) + '.png', img)


def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
    g = g[int(size) + 1, :]

    g = g/np.max(g)
    return g


def filter(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    filter = np.zeros(img.shape)
    center = int(img.shape[1]/2)
    w = int(img.shape[1]/2)
    g = gaussian_kernel(w)
    g = np.repeat(g.reshape(1,g.size),img.shape[0],axis=0)
    g = abs(1.0001-g)

    filter[:,center - w:center + w] = g[:,:-1]
    fshift = fshift*filter
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    a = np.fft.ifftshift(fshift)
    a = np.fft.ifft2(a)
    a = np.real(a)
    return np.uint8(a)


def heuristic_template(img,o1):
    heur = draw_template(img, 0, o1)
    heur = cv2.floodFill(heur,None, (int(heur.shape[1]/2), heur.shape[0]-1), 255)[1]
    heur = heur/255
    template = np.zeros(img.shape)
    template[template.shape[0]-heur.shape[0]:, o1:template.shape[1]-o1+1] = heur
    img = img*template
    return np.uint8(img)


def morph_op(img):
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph


def main():
    num_im = 50  # number of images to read from folder (308 files for training folder, 290 for testing)

    f = os.path.expanduser('~')+'/Downloads'   # directory where the dataset is located
    set = 'training'
    set_dirs(f,set)

    i = 0
    for n in sorted(glob.glob(dir_read+'/*.png')):
        im_rgb = cv2.imread(n)
        n = n[-13:-4]
        img = rgb_2_bw(im_rgb)

        edges = img_pre_process(img)
        edges = heuristic_template(edges, 0)
        morph = morph_op(edges)

        lanes = Hough_transform(morph, img, 50, 50, 50, n)
        if sum(sum(lanes)) == 0:
            continue
        template = draw_template(img,25,400)
        box,xy = templ_matching(lanes,template,img)

        edges = img_pre_process(box)
        final = Hough_transform(edges, img,70,50,10,n)
        if sum(sum(final)) == 0:
            continue
        align_lanes(im_rgb, final[:edges.shape[0],:edges.shape[1]], xy, n)
        print('Success for image #{}'.format(n))
        i += 1
        if i>num_im-1:
            break


main()