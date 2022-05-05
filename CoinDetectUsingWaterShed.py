import cv2
import numpy as np
import matplotlib.pyplot as plt

def coin_detect(im,borderSize,gap,minr,maxr,n):
    # 1 - Original:
    orig = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # 2 - Thresholding:
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 3 - Closing:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # 4 - Distance Transform:
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

##    borderSize = 75
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

##    gap = 55
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    # plt.imshow(nxcor, cmap='gray')
    # plt.show()
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)

    th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
    # plt.imshow(peaks, cmap='gray')
    # plt.show()
    peaks8u = cv2.convertScaleAbs(peaks)
    origscaled = cv2.convertScaleAbs(orig)
    peaks = np.uint8(peaks)
    _,contours, hierarchy = cv2.findContours(peaks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # peakss8u = cv2.convertScaleAbs(peaks)    # to use as mask
    canvass = np.zeros(peaks.shape, np.uint8)
    # im = np.uint8(im)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, radius, _, radiusloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks[y:y+h, x:x+w])
        # cv2.circle(canvass, (int(radiusloc[0] + x), int(radiusloc[1] + y)), int(radius), (255, 255, 255), 1)
        # if(radius<maxr and radius>minr):
        if (n==1):
            if (radius < maxr and radius > minr):
                cv2.circle(canvass, (int(radiusloc[0] + x), int(radiusloc[1] + y)), int(radius), (255, 255, 255), -1)

        if (n == 2):
            canvass2 = np.zeros(peaks.shape, np.uint8)
            if(radius<maxr and radius>minr):
                cv2.circle(canvass, (int(radiusloc[0] + x), int(radiusloc[1] + y)), int(radius), (255, 255, 255), -1)
                if (h == 18 and w == 16):
                    cv2.circle(canvass2, (int(radiusloc[0]+x), int(radiusloc[1]+y)), int(radius), (255, 255, 255), -1)
                    # print(radius,w,h)
                canvass = cv2.subtract(canvass,canvass2)
        if (n==3):
            canvass2 = np.zeros(peaks.shape, np.uint8)
            if (radius < maxr and radius > minr and h!=17):
                cv2.circle(canvass, (int(radiusloc[0] + x), int(radiusloc[1] + y)), int(radius), (255, 255, 255), -1)

            if (radius < 22 and radius > 18 and h == 18 and w == 16):
                cv2.circle(canvass2, (int(radiusloc[0] + x), int(radiusloc[1] + y)), int(radius), (255, 255, 255), -1)
            canvass = cv2.add(canvass, canvass2)

        if (n == 4):
            if (radius < maxr and radius > minr and h==17):
                cv2.circle(canvass, (int(radiusloc[0] + x), int(radiusloc[1] + y)), int(radius), (255, 255, 255), -1)

    result = cv2.bitwise_and((np.uint8(orig)), (np.uint8(orig)), mask=(np.uint8(canvass)))
    return result

# im = cv2.imread("C:\\Users\\Josephine\\Documents\\Water_shed\\BARYA_7.png")
im = cv2.imread("BARYA_1.png")
orig = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.imshow(orig),plt.title('Original')

plt.figure(2)
coin=coin_detect(im,18,5,15,18,1)
plt.imshow(coin),plt.title('25 CENTAVO')

plt.figure(3)
coin=coin_detect(im,18,5,18,22,2)
plt.imshow(coin),plt.title('ONE PESOS')

plt.figure(4)
coin=coin_detect(im,18,5,21.8,26,3)
plt.imshow(coin),plt.title('FIVE PESOS')

plt.figure(5)
coin=coin_detect(im,18,5,21.8,26,4)
plt.imshow(coin),plt.title('TEN PESOS')


plt.show()