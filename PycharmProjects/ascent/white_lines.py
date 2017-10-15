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


def img_pre_process(edges):
    k = 9
    k2 = 3
    edges = cv2.GaussianBlur(edges, (k, k), 0)
    #edges = cv2.Canny(edges, 150, 200,L2gradient=False)
    #edges = cv2.Sobel(edges, -1, 1, 0, ksize=3)
    edges = cv2.Sobel(edges, -1, k2-2, 0, ksize=k2)
    edges = cv2.Canny(edges, 150, 200,L2gradient=False)
    #edges = cv2.GaussianBlur(edges, (k, k), 0)
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


def draw_template(img,offset1,offset2,name):
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
    cv2.imwrite(dir + '/' + name + '.png', template)
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
    cv2.imwrite(dir + '/' + 'find_lane' + '.png', img)
    return lanes[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]], (top_left,bottom_right)


def align_lanes(img,edges,xy,n):
    highlight = np.zeros(img[:,:,0].shape)
    highlight[xy[0][1]:xy[1][1],xy[0][0]:xy[1][0]] = edges
    img[:, :, 0] = np.clip(img[:,:,0] - highlight*255, 0, None)
    img[:, :, 1] = np.clip(img[:,:,1] - highlight*255, 0, None)
    img[:, :, 2] = np.clip(img[:,:,2] + highlight*255, None, 255)
    #img[:, :, 0] = img[:,:,0] - highlight
    #img[:, :, 1] = img[:,:,1] - highlight
    #img[:, :, 2] = img[:,:,2] + highlight
    cv2.imwrite(dir + '/' + 'final_img_' + str(n).zfill(3) + '.png', img)


def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
    #g = g[int(size) + 1, int(size / 2) + 1:int(int(size) * 1.5) + 1]
    #g = g[int(size) + 1, int(size / 2) + 1:int(int(size) * 1.5) + 1]
    g = g[int(size) + 1, :]

    #g = g[int(size)+1,int(size/2)+1:int(size)+1]
    #g = np.concatenate((g,np.flip(g,axis=0)))
    g = g/np.max(g) #/ g.sum()
    #g = np.sqrt(g)
    return g


def filter(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    filter = np.zeros(img.shape)
    center = int(img.shape[1]/2)
    w = int(img.shape[1]/2)
    #w = 200

    #g = gaussian_kernel(img.shape[0],w*2)
    #g = gaussian_kernel(img.shape[0]/2, w)
    g = gaussian_kernel(w)
    g = np.repeat(g.reshape(1,g.size),img.shape[0],axis=0)
    g = abs(1.0001-g)

    #a = np.ones((w*2,img.shape[1]))

    #tx = img.shape[1]/2 - w
    #ty = 0
    #M = np.float32([[1, 0, tx], [0, 1, ty]])
    #g = cv2.warpAffine(g, M, (img.shape[1], img.shape[0]))

    #M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),45, 1)
    #g = cv2.warpAffine(g, M, (img.shape[1], img.shape[0]))
    #M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),-45, 1)
    #g2 = cv2.warpAffine(g, M, (img.shape[1], img.shape[0]))
    #g = (g1+g2)/2

    #filter[center-w:center+w,:] = g
    filter[:,center - w:center + w] = g[:,:-1]
    #filter = g
    fshift = fshift*filter
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    a = np.fft.ifftshift(fshift)
    a = np.fft.ifft2(a)
    a = np.real(a)

    #plt.subplot(211), plt.imshow(a, cmap='gray')
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(212), plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    #plt.show()

    return np.uint8(a)


def heuristic_template(img,o1):
    heur = draw_template(img, 0, o1,'heur')
    heur = cv2.floodFill(heur,None, (int(heur.shape[1]/2), heur.shape[0]-1), 255)[1]
    heur = heur/255
    template = np.zeros(img.shape)
    template[template.shape[0]-heur.shape[0]:, o1:template.shape[1]-o1+1] = heur
    img = img*template
    cv2.imwrite(dir + '/' + 'heur' + '.png', img)
    return np.uint8(img)


def main():
    images = np.arange(20)
    #images = [15]
    for n in images:
        im_rgb = load_im(n)
        img = rgb_2_bw(im_rgb)
        #img = filter(img)

        #while sum(edges) <
        #o1 = int(img.shape[1]/2)-100
        o1 = 00
        #lanes = np.zeros(img.shape)

        edges = img_pre_process(img)
        edges = heuristic_template(edges, o1)
        lanes = Hough_transform(edges, img, 50, 50, 50, 'lanes')

        #while sum(sum(lanes))/255 < 300:
        #    img = heuristic_template(img,o1)
        #    edges = img_pre_process(img)
            #lanes = Hough_transform(edges, img,80,20,15,'lanes')
            #lanes = Hough_transform(edges, img,50,50,50,'lanes')
            #lanes = Hough_transform(edges, img,50,50,50,'lanes')
        #    lanes = Hough_transform(edges, img,10,10,50,'lanes')
            #bounds = Hough_transform(edges, img,150,1,25,'bounds')
        #    print(sum(sum(lanes))/255, o1)
        #    o1 = o1 - 100

        #kernel = np.ones((1, 1), np.uint8)
        #erosion = cv2.morphologyEx(bounds, cv2.MORPH_CLOSE, kernel)
        #kernel = np.ones((3, 3), np.uint8)
        #blackhat = cv2.morphologyEx(erosion, cv2.MORPH_BLACKHAT, kernel)
        #cv2.imwrite(dir + '/' + 'bh' + '.png', blackhat)

        template = draw_template(img,25,400,'template')
        box,xy  = templ_matching(lanes,template,img)


        edges = img_pre_process(box)
        final = Hough_transform(edges, img,70,50,10,'final')
        #final = Hough_transform(edges, img,80,50,50,'final')
        #bounds = Hough_transform(edges, img,100,1,50,'bounds')
        align_lanes(im_rgb, final[:edges.shape[0],:edges.shape[1]], xy, n)
        print(n)






main()
plt.show()