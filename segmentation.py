import math
import glob
from cv2 import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from skimage import data,color,img_as_ubyte
from skimage.color import rgb2hsv
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
import os
# parameters

datadir = './for_bgst_data'
resultdir='./for_bgst_results'

sigma=2
threshold=0.03
rhoRes=1
thetaRes=math.pi/180
nLines=20
x_GK = 3
y_GK = 3

def ConvFilter(Igs, G):
    pad_width_row=(G.shape[0]-1)//2
    pad_width_col=(G.shape[1]-1)//2
    padded_image=np.pad(Igs,((pad_width_row,pad_width_row),(pad_width_col,pad_width_col)),'reflect')
    Iconv = np.zeros(Igs.shape)
    k=np.zeros(G.shape)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            k[i,j]=G[G.shape[0]-i-1,G.shape[1]-j-1]
    # kernel G inversing
    for y in range(Iconv.shape[1]):
        for x in range(Iconv.shape[0]):
            Iconv[x,y] = (k*padded_image[x:x+G.shape[0],y:y+G.shape[1]]).sum()
    # sum and multiplying
    return Iconv


def EdgeDetection(Igs, sigma):
    arrX=np.arange(math.trunc(x_GK/2)*(-1),math.ceil(x_GK/2),1)
    arrY=np.arange(math.trunc(y_GK/2)*(-1),math.ceil(y_GK/2),1)
    kernel_raw_X=np.exp((-arrX*arrX)/(2*sigma*sigma))
    kernel_raw_Y=np.exp((-arrY*arrY)/(2*sigma*sigma))
    kernel_X=kernel_raw_X/kernel_raw_X.sum()
    kernel_Y=kernel_raw_Y/kernel_raw_Y.sum()
    kernelX=np.array(kernel_X)
    kernelY=np.array(kernel_Y)
    G=np.zeros((x_GK,y_GK))
    for x in range(x_GK):
        for y in range(y_GK):
            G[x,y]=kernelX[x]*kernelY[y]
    SobelX_5=np.array([[-1,-2,0,2,1],[-2,-3,0,3,2],[-3,-5,0,5,3],[-2,-3,0,3,2],[-1,-2,0,2,1]]) #y pixel change significantly
    SobelY_5=np.array([[1,2,3,2,1],[2,3,5,3,2],[0,0,0,0,0],[-2,-3,-5,-3,-2],[-1,-2,-3,-2,-1]]) #x pixel change significantly
    Is=ConvFilter(Igs,G)
    Ix=ConvFilter(Is,SobelX_5)
    Iy=ConvFilter(Is,SobelY_5)
    Im=np.hypot(Ix,Iy)
    Io=np.arctan(Ix/Iy)
    #nms? or not?
    return Im, Io, Ix, Iy

def OrientationConfigure(Io):
    size = Io.size
    result = np.zeros(180)
    result2 = np.zeros(18)
    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
            angle = Io[i][j]/math.pi * 180
            angle = angle + 90
            angle = angle.astype(int)
            k = (angle-1)//10
            if(angle == 180): angle = 0
            result[angle] += 1
            result2[k] += 1
    result2 = result2/size   
    result = result/size
    return result, result2

def ColorRatio(Ig):
    size = Ig.size/3
    sumR = 0
    sumG = 0
    sumB = 0
    for i in range(Ig.shape[0]):
        for j in range(Ig.shape[1]):
            sumR += Ig[i][j][0]
            sumG += Ig[i][j][1]
            sumB += Ig[i][j][2]
    r = sumR/(sumR+sumG+sumB)
    g = sumG/(sumR+sumG+sumB)
    b = sumB/(sumR+sumG+sumB)
    return r,g,b

def ColorAvgGrad(Ig):
    k= np.gradient(Ig)
    _R = np.average(k[0])
    _G = np.average(k[1]) 
    _B = np.average(k[2])
    return _R,_G,_B

def ColorVariance(Ig):
    k = np.gradient(Ig)
    Vr = np.var(k[0])
    Vg = np.var(k[1])
    Vb = np.var(k[2])
    return Vr,Vg,Vb

def hueValueConfig(hsv_img):
    k = hsv_img[:,:,0]
    H = np.zeros(1000)
    for i in range(hsv_img.shape[0]):
        for j in range(hsv_img.shape[1]):
            index = k[i][j]*1000
            index = index.astype(int)
            H[index] += 1
    return H
def main():
    # read images
    print(datadir)
    
    for img_path in glob.glob(datadir+'/*.jpg'):
        print(img_path)
        #img = Image.open(img_path)
        #img_gray_scale = Image.open(img_path).convert("L")
        
        img = cv2.imread(img_path)
        cv2.GaussianBlur(img,(5,5),2.0)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)

        light_green = (200,255,200)
        strong_green = (0,120,0)
        mask = cv2.inRange(img_rgb,strong_green,light_green)
        result = cv2.bitwise_and(img,img,mask = mask)
        cv2.imwrite(img_path+'_1',result)
        Conv_hsv_Gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        result[mask == 255] = [255, 255, 255]

        cv2.imwrite(img_path+'_2',result)
        
        light_green = (30,20,20)
        dark_green = (70,250,250)

        mask = cv2.inRange(img_hsv,light_green,dark_green)
        result = cv2.bitwise_and(img,img,mask=mask)
        cv2.imwrite(img_path+'_3',result)
        
        Conv_hsv_Gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        result[mask == 255] = [255, 255, 255]

        cv2.imwrite(img_path+'_4',result)
        
        result = cv2.resize(img,(256,256))
        gray = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256,256), np.uint8)
        masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
        dst = cv2.bitwise_and(result, result, mask=mask)
        Conv_hsv_Gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        result[mask == 255] = [255, 255, 255]
        cv2.imwrite(img_path+'_5',result)
        
        sharp_img = cv2.createBackgroundSubtractorMOG2().apply(img)
        cv2.imwrite(img_path+'_6',sharp_img)

        pic_n = img.reshape(img.shape[0]*img.shape[1],img.shape[2])
        kmeans = KMeans(n_clusters= 2, random_state = 0).fit(pic_n)
        pic2show = kmeans.cluster_centers_[kmeans.labels_]
        cluster_pic = pic2show.reshape(img.shape[0],img.shape[1],img.shape[2])
        mask = np.zeros((img.shape[0],img.shape[1]),np.uint8)
        cluster_pic = cv2.bitwise_and(img,cluster_pic,mask=mask)_
        cv2.imwrite(img_path+'_8',cluster_pic)
        #Ig =  np.array(img)           
        #Ig_g = np.array(img_gray_scale)
        #Im, Io, Ix, Iy = EdgeDetection(Ig_g, sigma)
        #Io_configure1, Io_configure2 = OrientationConfigure(Io)
        #r,g,b = ColorRatio(Ig)
        #_R,_G,_B = ColorAvgGrad(Ig)
        #Vr,Vg,Vb = ColorVariance(Ig)
        #hsv_img = rgb2hsv(Ig)
        #H = hueValueConfig(hsv_img)



if __name__ == '__main__':
    main()
