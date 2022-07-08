#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:59:04 2022

@author: bryantchung
"""

#import packages
import cv2   
import imutils
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import sobel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    #i think like every like 50 iterations the code captures an image and adds it to the list
    while success:
        count += 1
        success, image = vidObj.read()
        if(count % 50 == 0):
            X_data.append(image)
        if count %1000 == 0:
            success = 0
            
        
    return X_data

#function for finding the closest X points near the snout 
def pointfinder(plane, index):
    minidx = []
    primeindex=list(chain.from_iterable(c[:, :, plane]))
    #gets list of desired coordinates (0 if looking right to left and 1 if looking top to bottom--yes this logic works)
    #print("list of prime coordinates")
    #print(primeindex)
    #referenceindex=list(chain.from_iterable(c[:, :, 1]))
    #print("list of reference coordinates")
    #print(referenceindex)
    primeindexuse=np.argsort(primeindex)
    #index=1 means finding top points
    if index==1:
        reverse=list(reversed(primeindexuse))
        for i in reverse[:10]:
                minidx.append(i)
    elif index==0:
        primeindexiter=list(primeindexuse)
        for j in primeindexiter[:10]:
            minidx.append(j)
    #print(minidx)
    pointslist=[]
    #for top to bottom 
    for index in minidx:
            #iterates through indices
# =============================================================================
#                     print("x coordinate")
#                     print(primeindexuse[index])
# =============================================================================
            use3=list(chain.from_iterable(c[:, :, :]))
            #gets x and y arrays?
            point = use3[index]
            #finds y coordinate associated with x coordinate (bye)
            pointslist.append(point)
            #print("full point")
            #print(point)
            #this prints the whole point
    #print("all points list")
    #print(pointslist)
    return pointslist
    
#for first four frames...
#proper results should be 176,176,175,175,174 for top
#proper results for bottom should be 71,71,72.73.75 or smth
#proper results for right should be 353,353,352,352,352
#proper results for left should be 316,316,317,317,317

#function for confirming location of snout using triangular analysis
def confirmsnout(snout):
    if snout == extRight:
        print("final snout designation")
        print("right")
        iterate = pointfinder(0, 1)
        for point in iterate:
             for point2 in iterate:
                if (point[1]>snout[1] and point2[1]<snout[1]) or (point[1]<snout[1] and point2[1]>snout[1]):
                #so if above or below the "snout" area -- same for R and L 
                    #print("right")
                    angle=findangle(point, point2)
                    if angle is not None:
                        print(angle)
    elif snout==extLeft:
        print("final snout designation")
        print("left")
        iterate = pointfinder(0, 0)
        for point in iterate:
            #print(point)
            for point2 in iterate:
                #print(point2)
                if (point[1]>snout[1] and point2[1]<snout[1]) or (point[1]<snout[1] and point2[1]>snout[1]):
                    #print("left")
                    angle=findangle(point, point2)
                    if angle is not None:
                        print(angle)
    elif snout==extBot:
        print("final snout designation")
        print("bot")
        iterate = pointfinder(1, 1)
        for point in iterate:
            #print(point)
            for point2 in iterate:
                #print(point2)
                if (point[0]>snout[0] and point2[0]<snout[0]) or (point[0]<snout[0] and point2[0]>snout[0]):
                #so if to left or to right -- same for T & B
# =============================================================================
#                     print("chosenpoints")
#                     print(point)
#                     print(point2)
# =============================================================================
                    angle=findangle(point, point2)
                    if angle is not None:
                        print(angle)
    elif snout==extTop:
        print("final snout designation")
        print("top")
        iterate = pointfinder(1, 0)
        for point in iterate:
             for point2 in iterate:
                if (point[0]>snout[0] and point2[0]<snout[0]) or (point[0]<snout[0] and point2[0]>snout[0]):
                    #print("top")
                    angle=findangle(point, point2)
                    if angle is not None:
                        print(angle)

#function for calculating the angle using dot product stuff 
def findangle(point, point2):
    a=point
    b=snout
    c = point2
    #shuffling around doesn't rlly change anything besides angle values 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    radangle = np.arccos(cosine_angle)
    degangle=radangle*57.2958
    #converts to degrees
    #print(degangle)
    if degangle<= 90:
        #print("it works!")
# =============================================================================
#         print(a)
#         print(b)
#         print(c)
#         print(ba)
#         print(bc)
#         print(cosine_angle)
# =============================================================================
        return degangle

#THIS IS THE ACTUAL CODE NOW
#iterates through images? 
if __name__ == '__main__':
    X_data_array = FrameCapture("/Users/bryantchung/Downloads/ExtraCurricular/UCI Summer 2022/mousevid.wmv")
    left1images = []
    left2images = []
    right3images = []
    right4images = []
    #iterates through the images
    for i in range(0,len(X_data_array)):
        img = X_data_array[i]
        # load image
    
        # start vertical devide image
        height = img.shape[0]
        width = img.shape[1]
        # Cut the image in half
        width_cutoff = width // 2
        left1 = img[:, :width_cutoff]
        right1 = img[:, width_cutoff:]
        # finish vertical devide image
        
        #rotate image LEFT1 to 90 CLOCKWISE -- why?
        img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
        # start vertical devide image
        height = img.shape[0]
        width = img.shape[1]
        # Cut the image in half
        width_cutoff = width // 2
        l2 = img[:, :width_cutoff]
        l1 = img[:, width_cutoff:]
        # finish vertical devide image
        #rotate image to 90 COUNTERCLOCKWISE
        l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #save
        left2images.append(l2)
        #rotate image to 90 COUNTERCLOCKWISE
        l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #save
        l1 = l1[80:360, 200:600]
        left1images.append(l1)
        
        #rotate image RIGHT1 to 90 CLOCKWISE
        img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
        # start vertical devide image
        height = img.shape[0]
        width = img.shape[1]
        # Cut the image in half
        width_cutoff = width // 2
        r4 = img[:, :width_cutoff]
        r3 = img[:, width_cutoff:]
        # finish vertical devide image
        #rotate image to 90 COUNTERCLOCKWISE
        r4 = cv2.rotate(r4, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #save
        right4images.append(r4)
        #rotate image to 90 COUNTERCLOCKWISE
        r3 = cv2.rotate(r3, cv2.ROTATE_90_COUNTERCLOCKWISE)
        right3images.append(r3)
        
    
   #iterates through four images (so all analysis code for each frame is embedded in this for loop)
    for i in range(0,4):
        
        dst = left1images[i]
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image -- think by color?, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        img_not = cv2.bitwise_not(thresh)
        cnts = cv2.findContours(img_not.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        #print(cnts)
        #okay yeah this is very precise
        cnts = imutils.grab_contours(cnts)
# =============================================================================
#         print("contours")
#         print(cnts)
# =============================================================================
        c = max(cnts, key=cv2.contourArea)

        #print("max contour")
        #print(c)

        #nested lists (two additional layers?)
        #gets the biggest contour
        #there's no difference? -- first one prints an array you cant use?-- okay just gets the biggest contour 
        #(so the rat contour) -- issue is that the tail sometimes distinguished as a different object
    
        # determine the most extreme points along the contour 
        #first column/row in last dimension 
        #so 0 = horizontal
        #ah no i get it the first two :: say get all of the lists and then the 0 means get the x value
        #[0] = vertical height is 0?, no axis size is just 0 just ignore that
        #tuple = ordered list that you can't change 
        #basically have to manipulate so gives a cluster of points in that region     
  
        
        #yields the entire extreme coordinate
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        #print("left coordinates")
        pointfinder(0, 0)
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        #print("right coordinates")
        pointfinder(0, 1)
        #1 = vertical
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        #print("top coordinates")
        pointfinder(1,1)
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        #print("bottom coordinates")
        pointfinder(1, 0)

# =============================================================================
#         print(extLeft[0])
#         #print("extright")
#         print(extRight[0])
#         print("exttop")
#         print(extTop[1])
#         print("extbot")
#         print(extBot[1])
# =============================================================================
        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        #yea its just a singular thick point
        cv2.drawContours(dst, [c], -1, (0, 255, 255), 2)
        cv2.circle(dst, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(dst, extRight, 8, (0, 255, 0), -1)
        cv2.circle(dst, extTop, 8, (255, 0, 0), -1)
        cv2.circle(dst, extBot, 8, (255, 255, 0), -1)
        
        
        # show the output image
        cv2.imshow("thresh", thresh)
        cv2.imshow("opp", img_not)
        cv2.imshow("Image", dst)
        
        #print(dst.shape)
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
            if region.area >= 500:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                #yea what is going on with this
                print("DIMENSIONS")
                print(minc)
                #this is min x value
                print(maxc)
                #this is max x value
                print(minr)
                #this is min y value
                print(maxr)
                #this is max y value
                #isnt this code supposed to exclude the tail 
        
        #final snout calculation
        snout = (0, 0)
        if abs(extLeft[0] - minc) > 20:
            snout = extRight
            print("RIGHT")
        if abs(extRight[0] - maxc) > 20:
            snout = extLeft
            print("LEFT")
        if abs(extTop[0] - minr) > 20:
            snout = extBot
            print("BOT")
        if abs(extBot[0] - maxr) > 20:
            snout = extTop
            print("TOP")
        print("snout")
        print(snout)
        #this algorithm is faulty since it can register as both bot/top and right/left
        #kk i fixed this with elif -- no this changes the angle and makes it weird
        #first image...top would be the most accurate...but it basically designates that it's top bc luck 
        #the thing is triangular analysis would fail if tail included bc tail would also be a sharp angle 
        #second image...bot turned out to be better..again luck of the draw
        #this one is less clear...but it shouldn't rlly matter for triangular analysis anyways 
        #actually bot is best... triangular analysis should also confirm this? 
        #angle would be too wide if chosen as most left...actually  maybe it wont since two similar circular shapes
        #third image is just inaccurate
        #fourth image should be left? -- yea but triangular analysis confirmed this was indeed not the snout
        #probs better to have an algorithm that sees which distance is the shortest
        #or maybe go for a value lower than 20
        #print("confirmanalysis")
        confirmsnout(snout)       

#triangular analysis for snout

a = np.array([155, 200])
b = np.array([140, 210]) #b has to be the snout area 
c = np.array([155, 225])
#78.69 degrees
ba = a - b
bc = c - b
# =============================================================================
# print(ba)
# print(bc)
# =============================================================================

cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#print(cosine_angle)
angle = np.arccos(cosine_angle)
#print(angle)


print("target angle")
print(angle*57.2958)
        

#print(snout)

ax.set_axis_off()
plt.tight_layout()
plt.show()
cv2.waitKey(1)
        
        
