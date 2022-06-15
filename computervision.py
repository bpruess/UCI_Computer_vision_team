#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 10:08:00 2022

@author: bryantchung
"""

import cv2
import math
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import skimage.io
import skimage.color 
import skimage.filters 

#imports image onto matplot plot
img = mpimg.imread('/Users/bryantchung/Downloads/ExtraCurricular/UCI Summer 2022/True/l-0-frame3338.png')
imgplot = plt.imshow(img)
plt.colorbar()

#turns figure gray, blurs it, and adds axes
gray_image = skimage.color.rgb2gray(img)
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap="gray")
plt.show()

#creats labeled histogram for frequency of each pixel
histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))

fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
#plt.show()

#thresholding
t=0.29
binary_mask = blurred_image < t

#masking?
fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap="gray")
plt.show()

selection = img.copy()
selection[~binary_mask] = 0

fig, ax = plt.subplots()
plt.imshow(selection)

# =============================================================================
# #gets rid of axes
# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#             hspace = 0, wspace = 0)
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.savefig("thresholdimage.pdf", bbox_inches = 'tight',
#     pad_inches = 0)
# plt.show()
# =============================================================================

#loads thresholded image from matplot into open CV
img = cv2.imread('/Users/bryantchung/Downloads/Random/Screenshots/Screen Shot 2022-06-03 at 10.58.21 AM.png')
#this works

# converts to grayscale --> black and white (contours distinguishes white objects from black background)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#running the contours (find outline of pixels of similar properties)
#chain_approx_none stores all boundary points 
contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


#prints number of points in outline and the coordinates
print('number of points: ',len(contours))

print('')

# =============================================================================
# #test to see what the contour points are like 
# # list contour points (consecutive points?)
# for pt in contours:
#         print(pt)
#         print("pt")
#         #pt is like 4 sets of consecutive points 
#         for inner in pt: 
#             print (inner)
#             print("inner")
#             print(inner[0][0])
#             print(inner[0][1])
#             #inners are the actual coordinates 
#             #inners are nested lists though but each represents one point 
# =============================================================================

# saves resulting images
cv2.imwrite('contours.png',im_bw)

#for checking axis labels in matplot 
image = cv2.imread('contours.png')
  
# convert color image into grayscale image -- unnecessary? 
img1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  
# plot that grayscale image with Matplotlib
# cmap stands for colormap
plt.imshow(img1, cmap='gray')
  
# display that image
plt.show()

# =============================================================================
# # show thresh and contour   
# #think dots are black and yes the contour works 
# cv2.drawContours(im_bw, contours, -1, (0,255,0), 3)
# cv2.imshow("contours", im_bw)
# #waits until a key is pressed
# cv2.waitKey(0)
# #destroys the window showing image
# cv2.destroyAllWindows()
# =============================================================================
            
#could use ear coordinates to supplement and pinpoint location of snout but its not cleary present in all frames
#vertical distance constraint for diameter 
for pt in contours:
        #pt is like 4 sets of consecutive points 
        for inner in pt: 
            innerx = inner[0][0]
            innery=inner[0][1]
# =============================================================================
#             print(innerx)
#             print(innery)
# =============================================================================
            #some zeroes but for the most part integer values
            for ptiterate in contours: 
                  for inneriterate in pt: 
                      #initially normal numbers and then towards the end a bunch of really small numbers (0,1,2, & 3)
                      #i think basically one (0,0) coordinate messes everything up
                      inneriteratex=inneriterate[0][0]
                      inneriteratey=inneriterate[0][1]
                      #this also doesn't work if rats are in different positions
                      if inneriteratex - innery >= 155:
                          #this is a really cheap solution...what if for false images rats whole body is in frame and snout is on edges?
# =============================================================================
#                           if (inneriteratex>=50) and (innerx >= 50):
#                               if (inneriteratey>=50) and (innery) >=50:
# =============================================================================
                                  distance = math.sqrt((inneriteratex-innerx)**2 + (inneriteratey-innery)**2)
        # =============================================================================
        #                           print(distance<=168)
        #                           print(distance>= 158)
        # =============================================================================
                                  if (distance<=168) and (distance>=158):
                                    print("yes")
                                    print(distance)
                                    print(innerx)
                                    print(innery)
                                    print(inneriterate[0][0])
                                    print(inneriterate[0][1])
                                 

                                
# saves resulting images
cv2.imwrite('contours.png',im_bw)

#for checking axis labels in matplot 
image = cv2.imread('contours.png')
  
# convert color image into grayscale image -- unnecessary? 
img1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  
# plot that grayscale image with Matplotlib
# cmap stands for colormap
plt.imshow(img1, cmap='gray')
  
# display that image
plt.show()

# =============================================================================
# # show thresh and contour   
# #think dots are black and yes the contour works 
# cv2.drawContours(im_bw, contours, -1, (0,255,0), 3)
# cv2.imshow("contours", im_bw)
# #waits until a key is pressed
# cv2.waitKey(0)
# #destroys the window showing image
# cv2.destroyAllWindows()
# =============================================================================
    
    
