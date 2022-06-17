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

def FrameCapture(path):
      
    vidObj = cv2.VideoCapture(path)
  
    count = 0
  
    success = 1
    X_data = []
  
    while success:
        count += 1
        success, image = vidObj.read()
        if(count % 50 == 0):
            X_data.append(image)
        if count %1000 == 0:
            success = 0
            
        
    return X_data
if __name__ == '__main__':
    X_data_array = FrameCapture("pyvideos/mousevid.wmv")
    left1images = []
    left2images = []
    right3images = []
    right4images = []
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
        
        #rotate image LEFT1 to 90 CLOCKWISE
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
        
        
    
   
    for i in range(0,4):
        
        dst = left1images[i]
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        img_not = cv2.bitwise_not(thresh)
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(img_not.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        print(extLeft[0])
        print(extRight[0])
        print(extTop[1])
        print(extBot[1])
        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        cv2.drawContours(dst, [c], -1, (0, 255, 255), 2)
        cv2.circle(dst, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(dst, extRight, 8, (0, 255, 0), -1)
        cv2.circle(dst, extTop, 8, (255, 0, 0), -1)
        cv2.circle(dst, extBot, 8, (255, 255, 0), -1)
        
        
        # show the output image
        cv2.imshow("thresh", thresh)
        cv2.imshow("opp", img_not)
        cv2.imshow("Image", dst)
        
        print(dst.shape)
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
                print(minc)
                print(maxc)
                print(minr)
                print(maxr)
        
        #final snout calculation
        snout = (0, 0)
        if abs(extLeft[0] - minc) > 20:
            snout = extRight
        if abs(extRight[0] - maxc) > 20:
            snout = extLeft
        if abs(extTop[0] - minr) > 20:
            snout = extBot
        if abs(extBot[0] - maxr) > 20:
            snout = extTop
        
        print(snout)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    cv2.waitKey(1)
        
        