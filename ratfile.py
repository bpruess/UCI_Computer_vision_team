#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:59:04 2022

@author: bryantchung
"""

#import packages (mostly for numerical analysis--triangular analysis and displaying the plots)
import cv2   
import imutils
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import sobel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#packages for cropping the video and doing image analysis 
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from itertools import chain

#function for implementing video 
def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
  
    count = 0
  
    success = 1
    X_data = []
    while success:
        count += 1
        success, image = vidObj.read()
        #every 50 frames puts into array--can lower number to include more frames 
        if(count % 50 == 0):
            #check 40th frame (snout at top, tail same)
            X_data.append(image)
        if count %1000 == 0:
            success = 0
            
        
    return X_data

#function for finding the closest X points near the snout 
def pointfinder(plane, index):
    minidx = []
    primeindex=list(chain.from_iterable(c[:, :, plane]))
    #gets list of desired coordinates (0 if looking right to left and 1 if looking top to bottom)
    #gets indices of points ordered from lowest to highest
    primeindexuse=np.argsort(primeindex)
    #index=1 means finding highest value points
    if index==1:
        reverse=list(reversed(primeindexuse))
        #gets the indices from highest to lowest value 
        for i in reverse[:20]:
                minidx.append(i)
    #index=0 means finding lowest value points
    elif index==0:
        primeindexiter=list(primeindexuse)
        #gets the indices from lowest to highest value
        for j in primeindexiter[:20]:
            minidx.append(j)
    pointslist=[]
    #for top to bottom 
    for index in minidx:
            #iterates through indices
            #gets x and y arrays
            use3=list(chain.from_iterable(c[:, :, :]))
            
            #finds y coordinate associated with x coordinate (index)
            point = use3[index]
            #adds the point
            pointslist.append(point)
    return pointslist

#function for confirming location of snout using triangular analysis
def confirmsnout(snout):
        none=0
        if (snout[0] == extRight[0]) and (snout[1] == extRight[1]):
            print("final snout designation")
            print("right")
            
            iterate = pointfinder(0, 1)
            for point in iterate:
                 for point2 in iterate:
                     #iterates through each point to check if above or below the "snout" area)
                    if (point[1]>snout[1] and point2[1]<snout[1]) or (point[1]<snout[1] and point2[1]>snout[1]):
                        angle=findangle(point, point2)
                        if angle is None:
                            none+=1
        elif (snout[0] == extLeft[0]) and (snout[1] == extLeft[1]):
            print("final snout designation")
            print("left")
            iterate = pointfinder(0, 0)
            for point in iterate:
                for point2 in iterate:
                    #iterates through code to check if above or below the "snout" area)
                    if (point[1]>snout[1] and point2[1]<snout[1]) or (point[1]<snout[1] and point2[1]>snout[1]):
                        #print("left")
                        angle=findangle(point, point2)
                        if angle is None:
                            none+=1
        elif (snout[0] == extBot[0]) and (snout[1] == extBot[1]):
            print("final snout designation")
            print("bot")
            iterate = pointfinder(1, 1)
            for point in iterate:
                for point2 in iterate:
                    #iterates through code to check if to right or to left of the "snout" area)
                    if (point[0]>snout[0] and point2[0]<snout[0]) or (point[0]<snout[0] and point2[0]>snout[0]):
                    #so if to left or to right -- same for T & B
                        angle=findangle(point, point2)
                        if angle is None:
                            none+=1
        elif (snout[0] == extTop[0]) and (snout[1] == extTop[1]):
            print("final snout designation")
            print("top")
            iterate = pointfinder(1, 0)
            for point in iterate:
                 for point2 in iterate:
                      #iterates through code to check if to right or to left of the "snout" area)
                    if (point[0]>snout[0] and point2[0]<snout[0]) or (point[0]<snout[0] and point2[0]>snout[0]):
                        #print("top")
                        angle=findangle(point, point2)
                        if angle is None:
                            none+=1
        #if not registered as extreme just not applicable 
        if none==0:
            return
        #write a code to correct it if it is incorrect 
            

#function for calculating the angle using dot product stuff 
def findangle(point, point2):
    a=point
    b=snout
    c = point2
    #shuffling around what is assigned to a,b, and cdoesn't rlly change anything besides angle values 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    radangle = np.arccos(cosine_angle)
    degangle=radangle*57.2958
    #converts to degrees
    return degangle

#THIS IS THE ACTUAL CODE NOW FOR RUNNING THE WHOLE ANALYSIS
#iterates through images? 
if __name__ == '__main__':
    X_data_array = FrameCapture("/Users/bryantchung/Downloads/ExtraCurricular/UCI Summer 2022/mousevid.wmv")
    #top left i believe 
    left1images = []
    fincoords = []

    for i in range(0,len(X_data_array)):
        img = X_data_array[i]
        img = img[100:400, 200:650]
        left1images.append(img)
        

        
    
   #iterates through four images (so all analysis code for each frame is embedded in this for loop)
    j=0
    for i in range(0,20):
        j+=1
        
        dst = left1images[i]
        #only analyzes one of the left boxes rn?
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image -- think by color?, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        img_not = cv2.bitwise_not(thresh)
        #prints points along perimeter of rat
        cnts = cv2.findContours(img_not.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        #gets the biggest contour
    
        # determine the most extreme points along the contour  
  
        
        #yields the entire extreme coordinate
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        pointfinder(0, 0)
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        pointfinder(0, 1)
        #(0,1) = top left
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        pointfinder(1,1)
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        #^^ these should all not be outside of the main body area 
        pointfinder(1, 0)

        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        #yeah its just a singular thick point

        cv2.drawContours(dst, [c], -1, (0, 255, 255), 2)
        
        
        # show the output image
        cv2.imshow("thresh", thresh)
        cv2.imshow("opp", img_not)
        cv2.imshow("Image", dst)
        
        #isolating rats by subtracting median
        medio = np.median(left1images, 0)
        plt.imshow(medio.astype(np.uint8))
        plt.imshow(np.abs(dst-medio).astype(np.uint8))
        #Sobel filter
        image = np.abs(dst-medio).astype(np.uint8)
        image = image[:,:,0]
        image_edge = sobel(image)
        gx = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
        gy = gx.T
        image_shape = np.shape(image)
        output_arrayx = np.zeros([image_shape[0], image_shape[1]])
        output_arrayy = np.zeros([image_shape[0], image_shape[1]])
        for y in range (2,image_shape[0]-1):
            for x in range(2,image_shape[1]-1):
                image_to_be_kerneled = image[y-2:y+1,x-2:x+1]
                output_arrayx[y,x] = np.sum(gx*image_to_be_kerneled)
                output_arrayy[y,x] = np.sum(gy*image_to_be_kerneled)
        plt.imshow(np.sqrt(output_arrayx**2+output_arrayy**2))
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))
        
        cleared = clear_border(bw)
        
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
        
        for region in regionprops(label_image):
            xcoor = 0
            ycoor = 0
            if region.area >= 100:
                print("big enough")
                #so if too small and scrunched up doesnt even register?
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                #add rectangle border of entire rat 
                print("DIMENSIONS for " + str(j) + " image")
                print("minx")
                print(minc)
                #this is min x value
                print("maxx")
                print(maxc)
                #this is max x value
                print("miny")
                print(minr)
                #this is min y value
                print("maxy")
                print(maxr)
                #this is max y value
                #isnt this code supposed to exclude the tail so if too far needs to be butt area since big tail area
                #these should exclude the tail i think 
        #final snout calculation
        print("------------------------------")
        print(extLeft[0])
        #left x coordinate
        print(extRight[0])
        #right x coordinate
        print(extTop[1])
        #top y coordinate
        print(extBot[1])
        #bottom y coordinate
        print("------------------------------")
        snout = (0, 0)
        #just placeholder
        extremes=[]
        differences = [abs(extLeft[0] - minc), abs(extRight[0] - maxc), abs(extTop[1] - minr), abs(extBot[1] - maxr)]
        print(differences)
        differences.sort()
        if abs(extLeft[0] - minc) == differences[3]:
        #checks for the largest difference 
            snout=extRight
            print("RIGHT")
        elif abs(extRight[0] - maxc) == differences[3]:
             snout = extLeft
             print("LEFT") 
        elif abs(extTop[1] - minr) == differences[3]:
             snout = extBot
             print("BOT")
        elif abs(extBot[1] - maxr) == differences[3]:
             snout = extTop
             print("TOP")
        print("snout")
        print(snout)
        #add a dot for the snout 
        fincoords.append(snout)

  
        confirmsnout(snout)    

#triangular analysis for snout

a = np.array([155, 200])
b = np.array([140, 210]) #b has to be the snout area 
c = np.array([155, 225])
#78.69 degrees
ba = a - b
bc = c - b

cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle = np.arccos(cosine_angle)

        
#displays all the graphs  
ax.set_axis_off()
plt.tight_layout()
plt.show()
#print(fincoords)
cv2.waitKey(1)
        
        
        
