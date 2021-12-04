from PIL import Image, ImageOps
import numpy as np
import os
import csv
import random
import math
import glob
import imageio
from cv2 import imread
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data,color,img_as_ubyte
from skimage.color import rgb2hsv
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.filters import threshold_otsu

#Load data
datadir = './classification'
rembgdir = '/shine_muscat_rembg/'
sigma=2
threshold=0.03
rhoRes=1
thetaRes=math.pi/180
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
    Io=Ix.copy()
    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
            if(Iy[i][j] < 0.01 and Iy[i][j] > -0.01 ):
                Io[i,j] = math.pi/2
                continue
            Io[i,j]=np.arctan(Ix[i][j]/Iy[i][j])
    return Io, Ix, Iy

def OrientationConfigure(Io):
    size = Io.size
    #result = np.zeros(180)
    n = 10
    result2 = np.zeros(n)
    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
            angle = Io[i][j]/math.pi * 180
            angle = angle + 90
            angle = angle.astype(int)
            t = 180 // n
            k = (angle-1)//t
            if(k>n-1): k=n-1
            #if(angle == 180): angle = 0
            #result[angle] += 1
            result2[k] += 1
    result2 = result2/size   
    #result = result/size
    return result2

def read_shuffle_csv(file_name):
    f = open(file_name,'r',encoding='UTF8')
    rdr = csv.reader(f)
    Line = []
    for line in rdr:
        Line.append(line)
    f.close()
    random.shuffle(Line)
    
    return Line

def remove_trash(Line_original, Line_trash):
    Line = []
    
    for i in range(len(Line_trash)):
        Line_trash[i] = Line_trash[i][0] + '.jpg'
    
    for i in Line_original:
        if (i[1] == ' accepted') and  (i[2] not in Line_trash) and (i[3] not in Line_trash) and (i[4] not in Line_trash):
            Line.append(i[2:])
    return Line

def jpg_to_png(Line):
    Line_png = []
    for line in Line:
        line_png = [0,0,0,0,0,0,0,0]
        line_png[0] = line[0][:-3]+'png'
        line_png[1] = line[1][:-3]+'png'
        line_png[2] = line[2][:-3]+'png'
        line_png[3:] = line[3:]
        Line_png.append(line_png)
    return Line_png

def getIO(Ig_g):
    Io,Ix,Iy = EdgeDetection(Ig_g,sigma)
    result = OrientationConfigure(Io)
    return result
        
def getRGB(Ig):
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

def gradient(img, x = True, y = True):
    f1 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    f2 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]]).T
    vert_gradient =ndimage.correlate(img, f1)
    horz_gradient =ndimage.correlate(img, f2)
    if x:
        return (horz_gradient)
    else:
        return (vert_gradient)

def ColorAvgGrad(Ig):
    k= np.gradient(Ig)
    Rx = gradient(Ig[:,:,0], y = False)
    Ry = gradient(Ig[:,:,0], x = False)
    Gx = gradient(Ig[:,:,1], y = False)
    Gy = gradient(Ig[:,:,1], x = False)
    Bx = gradient(Ig[:,:,2], y = False)
    By = gradient(Ig[:,:,2], x = False)
    gradR = np.sqrt(np.square(Rx) + np.square(Ry))
    gradG = np.sqrt(np.square(Gx) + np.square(Gy))
    gradB = np.sqrt(np.square(Bx) + np.square(By))
    _R = np.average(gradR)
    _G = np.average(gradG)
    _B = np.average(gradB)
    Vr = np.var(gradR)
    Vg = np.var(gradG)
    Vb = np.var(gradB)
    
    return _R,_G,_B,Vr,Vg,Vb

def hueValueConfig(hsv_img):
    k = hsv_img[:,:,0]
    H = np.zeros(20)
    for i in range(hsv_img.shape[0]):
        for j in range(hsv_img.shape[1]):
            index = k[i][j]*20
            index = index.astype(int)
            H[index] += 1
    return H
def getavgRGB(Ig):
    sumR = 0
    sumG = 0
    sumB = 0
    numPix = 0
    for i in range(Ig.shape[0]):
        for j in range(Ig.shape[1]):
            if(Ig[i][j][0]==0 and Ig[i][j][1] == 0 and Ig[i][j][2] ==0 ):
                continue
            numPix += 1
            sumR += Ig[i][j][0]
            sumG += Ig[i][j][1]
            sumB += Ig[i][j][2]
    avgR = sumR/numPix/255
    avgG = sumG/numPix/255
    avgB = sumB/numPix/255
    return avgR,avgG,avgB
def main():
    Line_original = read_shuffle_csv('response.csv')
    Line_trash = read_shuffle_csv('trash.csv')
    Line = remove_trash(Line_original, Line_trash)
    Line_png = jpg_to_png(Line)
    new_csv = open("dataskit.csv","w",encoding ='UTF8')
    csv_writer = csv.writer(new_csv)
    title = ['FileDir','Io']
    csv_writer.writerow(title)
    input_dir = []
    for i in range(len(Line_png)):
        for j in range(3):
            input_entry = []
            eachdir = datadir+rembgdir+Line_png[i][j]
            if(os.path.isfile(eachdir)):
                input_entry.append(eachdir)
                img_gray_scale = Image.open(eachdir).convert("L")
                img_gray_scale = ImageOps.fit(img_gray_scale,(96,96))       
                Ig_g = np.array(img_gray_scale)
                Io = getIO(Ig_g)
                for k in range(10):
                    input_entry.append(Io[k])
                csv_writer.writerow(input_entry)
    new_csv2 = open("dataskit2.csv","w",encoding = 'UTF8')
    csv_writer2 = csv.writer(new_csv2)
    title = ['FileDir','R','G','B','_R','_G','_B','Vr','Vg','Vb']
    csv_writer2.writerow(title)
    for i in range(len(Line_png)):
        for j in range(3):
            input_entry = []
            eachdir = datadir+rembgdir+Line_png[i][j]
            if(os.path.isfile(eachdir)):
                input_entry.append(eachdir)
                img = Image.open(eachdir).convert('RGB')
                img = ImageOps.fit(img,(96,96))
                Ig = np.array(img)
                R, G, B = getRGB(Ig)
                _R,_G,_B,Vr,Vg,Vb = ColorAvgGrad(Ig)
                input_entry.append(R)
                input_entry.append(G)
                input_entry.append(B)
                input_entry.append(_R)
                input_entry.append(_G)
                input_entry.append(_B)
                input_entry.append(Vr)
                input_entry.append(Vg)
                input_entry.append(Vb)
                csv_writer2.writerow(input_entry)

    new_csv3 = open("dataskit3.csv","w",encoding = 'UTF8')
    csv_writer3 = csv.writer(new_csv3)
    title = ['FileDir','HSV_Hue']
    csv_writer3.writerow(title)
    for i in range(len(Line_png)):
        for j in range(3):
            input_entry = []
            eachdir = datadir+rembgdir+Line_png[i][j]
            if(os.path.isfile(eachdir)):
                input_entry.append(eachdir)
                img = Image.open(eachdir).convert('RGB')
                img = ImageOps.fit(img,(96,96))
                Ig = np.array(img)
                hsv_img = rgb2hsv(Ig)
                H = hueValueConfig(hsv_img)
                for k in range(20):
                    input_entry.append(H[k])
                csv_writer3.writerow(input_entry)
    new_csv4 = open("dataskit4.csv","w",encoding = 'UTF8')
    csv_writer4 = csv.writer(new_csv4)
    title = ['FileDir','avgR','avgG','avgB']
    csv_writer4.writerow(title)
    for i in range(len(Line_png)):
        for j in range(3):
            input_entry = []
            eachdir = datadir+rembgdir+Line_png[i][j]
            if(os.path.isfile(eachdir)):
                input_entry.append(eachdir)
                img = Image.open(eachdir).convert('RGB')
                img = ImageOps.fit(img,(96,96))
                Ig = np.array(img)
                avgR, avgG, avgB = getavgRGB(Ig)
                input_entry.append(avgR)
                input_entry.append(avgG)
                input_entry.append(avgB)
                csv_writer4.writerow(input_entry)

if __name__ == '__main__':
    main()
